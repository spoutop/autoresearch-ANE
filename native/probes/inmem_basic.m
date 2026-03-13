#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

// Generate a weight blob with proper header for ANE (fp16 random weights)
static NSData *buildWeightBlob(int ch) {
    NSUInteger wsize = (NSUInteger)ch * ch * 2;
    NSUInteger total = 128 + wsize;
    uint8_t *buf = calloc(total, 1);
    buf[0] = 1; buf[4] = 2;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE; buf[68] = 1;
    *(uint32_t*)(buf+72) = (uint32_t)wsize;
    *(uint32_t*)(buf+80) = 128;
    uint16_t *fp16 = (uint16_t*)(buf + 128);
    for (NSUInteger j = 0; j < (NSUInteger)ch * ch; j++)
        fp16[j] = (arc4random() & 0x03FF) | 0x2000;
    return [NSData dataWithBytesNoCopy:buf length:total freeWhenDone:YES];
}

// Generate MIL program text for a 1x1 conv (ch -> ch, spatial = sp)
static NSString *genMIL(int ch, int sp) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}})]\n{\n"];
    [m appendFormat:@"    func main<ios18>(tensor<fp32, [1, %d, 1, %d]> x) {\n", ch, sp];
    [m appendString:
        @"        string c_pad_type = const()[name = string(\"c_pad_type\"), val = string(\"valid\")];\n"
        @"        tensor<int32, [2]> c_strides = const()[name = string(\"c_strides\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        tensor<int32, [4]> c_pad = const()[name = string(\"c_pad\"), val = tensor<int32, [4]>([0, 0, 0, 0])];\n"
        @"        tensor<int32, [2]> c_dilations = const()[name = string(\"c_dilations\"), val = tensor<int32, [2]>([1, 1])];\n"
        @"        int32 c_groups = const()[name = string(\"c_groups\"), val = int32(1)];\n"
        @"        string to_fp16 = const()[name = string(\"to_fp16\"), val = string(\"fp16\")];\n"];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> x16 = cast(dtype = to_fp16, x = x)[name = string(\"cast_in\")];\n", ch, sp];
    [m appendFormat:@"        tensor<fp16, [%d, %d, 1, 1]> W = const()[name = string(\"W\"), val = tensor<fp16, [%d, %d, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights/weight.bin\"), offset = uint64(64)))];\n", ch, ch, ch, ch];
    [m appendFormat:@"        tensor<fp16, [1, %d, 1, %d]> y16 = conv(dilations = c_dilations, groups = c_groups, pad = c_pad, pad_type = c_pad_type, strides = c_strides, weight = W, x = x16)[name = string(\"conv\")];\n", ch, sp];
    [m appendString:@"        string to_fp32 = const()[name = string(\"to_fp32\"), val = string(\"fp32\")];\n"];
    [m appendFormat:@"        tensor<fp32, [1, %d, 1, %d]> y = cast(dtype = to_fp32, x = y16)[name = string(\"cast_out\")];\n", ch, sp];
    [m appendString:@"    } -> (y);\n}\n"];
    return m;
}

int main() {
    @autoreleasepool {
        mach_timebase_info(&g_tb);
        dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

        NSError *e = nil;
        int ch = 256, sp = 64;

        // Generate MIL and weights programmatically (no external .mlpackage needed)
        NSData *milData = [[genMIL(ch, sp) dataUsingEncoding:NSUTF8StringEncoding] copy];
        NSData *weightBlob = buildWeightBlob(ch);

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @0, @"data": weightBlob}
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);

        // Get the hex identifier to pre-populate the temp dir
        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];

        // Pre-create dir with MIL and weights
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        printf("Pre-created: %s\n", [tmpDir UTF8String]);

        // Compile
        printf("Compiling...\n");
        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        printf("compile: %s\n", ok ? "YES" : "NO");
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e=nil; }
        if (!ok) return 1;

        // Load
        printf("Loading...\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        printf("load: %s\n", ok ? "YES" : "NO");
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e=nil; }
        if (!ok) return 1;

        printf("state: %lu\n", ((NSUInteger(*)(id,SEL))objc_msgSend)(model, @selector(state)));

        // Create IO surfaces
        NSUInteger bytes = ch * sp * 4;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);

        // Evaluate
        printf("Evaluating...\n");
        ok = ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, req, &e);
        printf("evaluate: %s\n", ok ? "YES" : "NO");
        if (e) { printf("  err: %s\n", [[e description] UTF8String]); e=nil; }

        if (ok) {
            // Warmup
            for (int i = 0; i < 10; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &e);

            // Benchmark
            int iters = 100;
            uint64_t t0 = mach_absolute_time();
            for (int i = 0; i < iters; i++)
                ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                    model, @selector(evaluateWithQoS:options:request:error:),
                    21, @{}, req, &e);
            double ms = ticksToMs(mach_absolute_time() - t0) / iters;
            double gf = 2.0*ch*ch*sp/1e9;
            double tflops = gf / ms;

            printf("\n========================================\n");
            printf("IN-MEMORY ANE EXECUTION SUCCESSFUL!\n");
            printf("  %.3f ms/eval, %.2f TFLOPS\n", ms, tflops);
            printf("========================================\n");
        }

        // Cleanup
        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
            model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];
    }
    return 0;
}
