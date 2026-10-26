// CPU-buildable capability probe; execute only through gpuq.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main() {
  @autoreleasepool {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    if (!d) return 1;
    printf("{\"device\":\"%s\",\"dispatch\":%s,\"stage\":%s,\"complete\":true}\n",
      d.name.UTF8String,
      [d supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary] ? "true" : "false",
      [d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary] ? "true" : "false");
  }
}
