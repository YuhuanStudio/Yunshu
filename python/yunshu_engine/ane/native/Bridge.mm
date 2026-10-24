// Yunshu ctypes ABI. Private-API failures are reported as unavailable, never thrown across C.
#include "Program.hpp"
#include "Handoff.hpp"
#include <cstring>
#include <future>
#include <mutex>
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
  bool stopped = false;
  std::mutex mutex;
};
using SessionHandle = std::shared_ptr<Session>;
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
    if (s->stopped || !repeats || repeats > 10000) throw std::runtime_error("native ANE session unavailable");
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
    try { *milliseconds = future.get(); }
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
    if (s->stopped && s->handoff && !s->handoff->idle()) new SessionHandle(s);
    delete holder;
  });
}
}
