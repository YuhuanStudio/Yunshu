// # Upstream (Splash 1.4.0, Apache-2.0): runtime/AwakeClock.hpp
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
// Unless required by law or agreed in writing, distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for permissions and limitations under the License.
// Yunshu local adapter for the Splash-derived ANE layer (Apache-2.0).
#pragma once
#import <Metal/Metal.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <time.h>
namespace yunshu {
// Same awake-time semantics as Splash's AwakeClock.hpp.
struct AwakeClock {
  using duration = std::chrono::nanoseconds;
  using rep = duration::rep;
  using period = duration::period;
  using time_point = std::chrono::time_point<AwakeClock>;
  static constexpr bool is_steady = true;
  static time_point now() noexcept { return time_point(duration(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))); }
};
inline uint64_t alignUp(uint64_t value) {
  const uint64_t page = static_cast<uint64_t>(getpagesize());
  return (value + page - 1) / page * page;
}
namespace metal {
// Phase 1: the IOSurface owns the memory. No MLX buffer is wrapped yet.
struct MetalBuffer {
  void *address = nullptr;
  std::shared_ptr<void> owner;
  void *contents() const { return address; }
};
class SharedEvent {
  struct State {
    __strong id<MTLSharedEvent> event;
    __strong MTLSharedEventListener *listener;
    State(id<MTLDevice> device) : event([device newSharedEvent]),
      listener([[MTLSharedEventListener alloc] init]) {
      if (!event) throw std::runtime_error("Metal shared event unavailable");
    }
  };
  std::shared_ptr<State> state;
public:
  explicit SharedEvent(id<MTLDevice> device) : state(std::make_shared<State>(device)) {}
  void *nativeHandle() const { return (__bridge void *)state->event; }
  void signal(uint64_t value) const noexcept { state->event.signaledValue = value; }
  void notify(uint64_t value, std::function<void()> callback) const {
    [state->event notifyListener:state->listener atValue:value
      block:^(id<MTLSharedEvent>, uint64_t) { callback(); }];
  }
};
struct MetalBackend {
  MetalBuffer wrapSharedMemory(void *address, uint64_t, std::shared_ptr<void> owner, const char *) {
    return {address, std::move(owner)};
  }
  SharedEvent newSharedEvent() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) throw std::runtime_error("Metal device unavailable");
    return SharedEvent(device);
  }
};
}
}
