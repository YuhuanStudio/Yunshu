// # Upstream (inspired): MLX v0.32.3 (MIT), Metal encoder and DLPack contracts; own C ABI and watchdogs.
// Yunshu ctypes ABI. Private-API failures are reported as unavailable, never thrown across C.
#include "Program.hpp"
#include "Handoff.hpp"
#ifdef SPLASH_ANE_INSTRUMENTATION
#include "ProgramInstrumentation.hpp"
#endif
#include <cstring>
#include <future>
#include <mutex>
#ifdef YS_ANE_MLX
#include "mlx/backend/metal/device.h"
#include "mlx/stream.h"
#endif
using namespace yunshu;
namespace {
thread_local std::string lastError;
std::atomic<bool> serviceRetired = false;
struct Session {
  metal::MetalBackend backend;
  std::unique_ptr<ane::Program> program;
  std::vector<ane::Surface> inputs;
  ane::Surface output;
  std::optional<ane::Program::Binding> binding;
  std::unique_ptr<ane::Handoff> handoff;
  std::atomic<bool> stopped = false;
  std::atomic<bool> busy = false;
  std::mutex mutex;
};
using SessionHandle = std::shared_ptr<Session>;
// Retain an unreported request until it really settles, including after timeout.
// No poll blocks a thread; a permanently stalled service retains just its state.
void retainUntilIdle(SessionHandle session) {
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_SEC),
    dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
      if (session->handoff && !session->handoff->idle()) retainUntilIdle(session);
    });
}
template<class F> int protect(F body) noexcept {
  try {
    @autoreleasepool {
      @try { body(); }
      @catch (NSException *e) {
        throw std::runtime_error(std::string("native ANE raised: ") + e.description.UTF8String);
      }
    }
    lastError.clear(); return 1;
  }
  catch (const std::exception &e) { lastError = e.what(); }
  catch (...) { lastError = "unknown native ANE failure"; }
  return 0;
}
}
extern "C" {
const char *ys_ane_error() { return lastError.c_str(); }
#ifdef SPLASH_ANE_INSTRUMENTATION
int ys_ane_test_arm(const char *kind) {
  return protect([&] {
    ane::ProgramInstrumentation::Faults faults;
    std::string name(kind);
    if (name == "selector") faults.changedMethod = "+[_ANEModel modelAtURL:key:]";
    else if (name == "objc") faults.raiseAt = "evaluation";
    else if (name == "async") faults.failingEvaluation = 1;
    else if (name == "stalled") faults.stalledEvaluation = 1;
    else throw std::invalid_argument("unknown test fault");
    ane::ProgramInstrumentation::arm(std::move(faults));
  });
}
#endif

int ys_ane_available() { return protect([] { if (serviceRetired) throw std::runtime_error("native ANE service retired after failure"); if (auto why = ane::unavailable()) throw std::runtime_error(*why); }); }
// All surfaces are fp16, channel-major [channels, tokens], padded to 64 bytes per row.
void *ys_ane_create(const char *mil, const uint8_t *weights, size_t size, const char *cache,
                    const uint32_t *heights, const uint32_t *widths, uint32_t count,
                    uint32_t outHeight, uint32_t outWidth) {
  SessionHandle result;
  int ok = protect([&] {
    if (serviceRetired) throw std::runtime_error("native ANE service retired after failure");
    auto s = std::make_shared<Session>();
    if (!count || count > 32 || !outHeight || !outWidth) throw std::invalid_argument("invalid surface dimensions");
    for (uint32_t i = 0; i < count; ++i) {
      if (!heights[i] || !widths[i]) throw std::invalid_argument("empty surface");
      s->inputs.push_back(ane::Surface::create(s->backend, heights[i], widths[i], ane::Surface::Element::Float16));
    }
    s->output = ane::Surface::create(s->backend, outHeight, outWidth, ane::Surface::Element::Float16);
    s->program = std::make_unique<ane::Program>(mil, std::span(weights, size),
      ane::Program::Limits{std::chrono::seconds(120), {}}, cache);
    uint32_t proc = s->program->procedure("main");
    s->binding = s->program->bind(proc, s->inputs, s->output);
    s->handoff = std::make_unique<ane::Handoff>(s->backend, std::chrono::seconds(2));
    result = std::move(s);
  });
  return ok ? new SessionHandle(std::move(result)) : nullptr;
}
void *ys_ane_retain(void *handle) {
  SessionHandle result;
  int ok = protect([&] {
    result = *static_cast<SessionHandle *>(handle);
    if (result->stopped || serviceRetired) throw std::runtime_error("native ANE session unavailable");
  });
  return ok ? new SessionHandle(std::move(result)) : nullptr;
}
int ys_ane_busy(void *handle) { return (*static_cast<SessionHandle *>(handle))->busy ? 1 : 0; }
void *ys_ane_data(void *handle, uint32_t index) {
  auto s = *static_cast<SessionHandle *>(handle);
  if (index > s->inputs.size()) return nullptr;
  return index == s->inputs.size() ? s->output.buffer.contents() : s->inputs[index].buffer.contents();
}
uint32_t ys_ane_stride(void *handle, uint32_t index) {
  auto s = *static_cast<SessionHandle *>(handle);
  if (index > s->inputs.size()) return 0;
  return index == s->inputs.size() ? s->output.strideBytes : s->inputs[index].strideBytes;
}
int ys_ane_run(void *handle, uint32_t repeats, double *milliseconds) {
  return protect([&] {
    auto s = *static_cast<SessionHandle *>(handle);
    std::lock_guard lock(s->mutex);
    if (s->stopped || serviceRetired || s->busy || !repeats || repeats > 10000) throw std::runtime_error("native ANE session unavailable");
    // Retain the session if evaluateWithModel itself stalls, not just its completion.
    auto promise = std::make_shared<std::promise<double>>();
    auto future = promise->get_future();
    const auto submitted = AwakeClock::now();
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
      @autoreleasepool {
        try {
          auto start = submitted;
          for (uint32_t i = 0; i < repeats; ++i) {
            auto ready = s->handoff->next(), done = s->handoff->next();
            s->handoff->queue(ready, done, [&](const auto &event, auto wait, auto signal, auto report) {
              s->program->enqueue(*s->binding, event, wait, signal, report);
            });
            s->handoff->event().signal(ready);
            if (!s->handoff->finish()) throw std::runtime_error(s->handoff->reason());
          }
          promise->set_value(std::chrono::duration<double, std::milli>(AwakeClock::now() - start).count() / repeats);
        } catch (...) { promise->set_exception(std::current_exception()); }
      }
    });
    if (future.wait_for(std::chrono::seconds(2 * repeats + 2)) != std::future_status::ready) {
      s->stopped = true;
      serviceRetired = true;
      throw std::runtime_error("native ANE evaluation timed out; session retired");
    }
    try { *milliseconds = future.get(); if (serviceRetired) throw std::runtime_error("native ANE service retired"); }
    catch (...) { s->stopped = true; serviceRetired = true; throw; }
  });
}
void ys_ane_destroy(void *handle) {
  protect([&] {
    auto holder = static_cast<SessionHandle *>(handle);
    auto s = *holder;
    // A request that never reports must not be unloaded while in flight.
    // Quarantine its model/surfaces until process exit. The process-wide breaker
    // prevents opening more sessions after an evaluation failure.
    if (s->stopped && s->handoff && !s->handoff->idle()) retainUntilIdle(s);
    delete holder;
  });
}
}

#ifdef YS_ANE_MLX
namespace {
struct MlxTicket : std::enable_shared_from_this<MlxTicket> {
  SessionHandle session;
  mlx::core::Stream stream;
  uint64_t ready, done;
  std::mutex mutex;
  std::condition_variable changed;
  bool settled = false, success = false, finalized = false, waitEncoded = false;
  std::string failure;
  MlxTicket(SessionHandle s, mlx::core::Stream stream)
      : session(std::move(s)), stream(stream), ready(session->handoff->next()), done(session->handoff->next()) {}
  void fail(std::string reason) {
    {
      std::lock_guard lock(mutex);
      if (failure.empty()) failure = std::move(reason);
      success = false;
      settled = true;
    }
    serviceRetired = true;
    session->stopped = true;
    session->handoff->retire("native MLX handoff failed");
    changed.notify_all();
  }
  void reported(bool ok) {
    if (!ok) { fail("native ANE evaluation failed"); return; }
    {
      std::lock_guard lock(mutex);
      if (failure.empty()) { success = true; settled = true; }
    }
    changed.notify_all();
  }
};
using TicketHandle = std::shared_ptr<MlxTicket>;
}
extern "C" {
void *ys_ane_mlx_begin(void *handle) {
  TicketHandle result;
  int ok = protect([&] {
    auto s = *static_cast<SessionHandle *>(handle);
    std::lock_guard lock(s->mutex);
    if (s->stopped || serviceRetired || s->busy.exchange(true))
      throw std::runtime_error("native ANE session unavailable or in flight");
    try {
      auto stream = mlx::core::default_stream(mlx::core::Device::gpu);
      auto ticket = std::make_shared<MlxTicket>(s, stream);
      result = ticket;
      auto event = s->handoff->event();
      // Watch ready independently of enqueue: even a stuck private evaluate
      // call must release a GPU wait within two seconds of its producer.
      event.notify(ticket->ready, [ticket] {
        dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC),
          dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
            bool settled;
            { std::lock_guard lock(ticket->mutex); settled = ticket->settled; }
            if (!settled) ticket->fail("ANE did not report within two seconds of GPU ready");
          });
      });
      dispatch_async(dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0), ^{
        @autoreleasepool {
          try {
            ticket->session->handoff->queue(ticket->ready, ticket->done,
              [ticket](const auto &event, auto ready, auto done, auto report) {
                ticket->session->program->enqueue(*ticket->session->binding, event, ready, done,
                  [ticket, report](bool success) { report(success); ticket->reported(success); });
              });
            if (ticket->session->handoff->retired()) ticket->fail(ticket->session->handoff->reason());
          } catch (const std::exception &e) { ticket->fail(e.what()); }
          catch (...) { ticket->fail("unknown ANE enqueue failure"); }
        }
      });
      // async_eval encodes/finalizes on this same MLX host thread. End its
      // encoder, signal the already-produced IOSurface, and submit without
      // waiting on the CPU. Commit now so GPU partial FFN and ANE can overlap.
      auto &encoder = mlx::core::metal::get_command_encoder(stream);
      encoder.end_encoding();
      id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)(void *)encoder.get_command_buffer();
      id<MTLSharedEvent> shared = (__bridge id<MTLSharedEvent>)event.nativeHandle();
      if (!command || !shared) throw std::runtime_error("MLX Metal command/event unavailable");
      [command encodeSignalEvent:shared value:ticket->ready];
      [command addCompletedHandler:^(id<MTLCommandBuffer> finished) {
        if (finished.status == MTLCommandBufferStatusError) ticket->fail("GPU producer command failed");
      }];
      encoder.commit();
    } catch (...) {
      if (result) result->fail("MLX handoff submission failed");
      s->stopped = true;
      serviceRetired = true;
      s->busy = false;
      throw;
    }
  });
  if (!ok) { if (result && !result->session->handoff->idle()) retainUntilIdle(result->session); return nullptr; }
  return new TicketHandle(std::move(result));
}
int ys_ane_mlx_wait(void *handle) {
  auto ticket = *static_cast<TicketHandle *>(handle);
  int ok = protect([&] {
    {
      std::lock_guard lock(ticket->mutex);
      if (ticket->waitEncoded) throw std::runtime_error("MLX wait already encoded");
      ticket->waitEncoded = true;
    }
    if (mlx::core::default_stream(mlx::core::Device::gpu) != ticket->stream)
      throw std::runtime_error("ANE handoff used from another MLX stream");
    auto &encoder = mlx::core::metal::get_command_encoder(ticket->stream);
    encoder.end_encoding();
    id<MTLCommandBuffer> command = (__bridge id<MTLCommandBuffer>)(void *)encoder.get_command_buffer();
    id<MTLSharedEvent> shared = (__bridge id<MTLSharedEvent>)ticket->session->handoff->event().nativeHandle();
    if (!command || !shared) throw std::runtime_error("MLX Metal command/event unavailable");
    [command encodeWaitForEvent:shared value:ticket->done];
  });
  if (!ok) ticket->fail("MLX consumer wait failed");
  return ok;
}
int ys_ane_mlx_finish(void *handle) {
  auto ticket = *static_cast<TicketHandle *>(handle);
  bool releaseBusy = false;
  int ok = protect([&] {
    if (ticket->finalized) return;
    {
      std::unique_lock lock(ticket->mutex);
      if (!ticket->changed.wait_for(lock, std::chrono::seconds(2), [&] { return ticket->settled; })) {
        lock.unlock(); ticket->fail("ANE completion did not settle"); lock.lock();
      }
      if (!ticket->success || serviceRetired)
        throw std::runtime_error(ticket->failure.empty() ? "native ANE service retired" : ticket->failure);
    }
    if (!ticket->session->handoff->finish()) throw std::runtime_error(ticket->session->handoff->reason());
    ticket->finalized = true;
    releaseBusy = true;
  });
  if (!ok) { ticket->fail("ANE completion validation failed"); releaseBusy = true; }
  if (releaseBusy) ticket->session->busy = false;
  return ok;
}
void ys_ane_mlx_destroy(void *handle) {
  protect([&] {
    auto holder = static_cast<TicketHandle *>(handle);
    auto ticket = *holder;
    if (!ticket->finalized) {
      ticket->fail("ANE ticket cancelled before validated completion");
      ticket->session->busy = false;
    }
    if (!ticket->session->handoff->idle()) retainUntilIdle(ticket->session);
    delete holder;
  });
}
}
#endif
