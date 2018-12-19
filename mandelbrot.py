#!/usr/bin/env python3
import time
import cv2
import numpy as np
import pyopencl as cl
import tkinter as tk
from PIL import Image, ImageTk
from six.moves import range

def calc_fractal_numpy(q, maxiter):
    output = np.zeros(q.shape, dtype=np.uint16)

    z = np.zeros(q.shape, np.complex)

    for it in range(maxiter):
        z = z*z + q
        done = np.greater(abs(z), 2.0)
        q = np.where(done, 0+0j, q)
        z = np.where(done, 0+0j, z)
        output = np.where(done, it, output)
    return output

def calc_fractal_opencl(q, maxiter):
    # List all the stuff in this computer
    platforms = cl.get_platforms()

    for platform in platforms:
        print("Found a device: {}".format(str(platform)))

    # Let's just go with device zero
    ctx = cl.Context(dev_type=cl.device_type.ALL,
                     properties=[(cl.context_properties.PLATFORM, platforms[1])])

    # Create a command queue on the device
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    # This is our main output (the image we're going to put on the screen)
    output = np.zeros(q.shape, dtype=np.uint16)

    # These are our buffers to hold data on the device
    q_opencl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=q)
    output_opencl = cl.Buffer(ctx, mf.WRITE_ONLY, output.nbytes)

    # This is our OpenCL kernel
    prg = cl.Program(ctx, """
    #pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable
    __kernel void mandelbrot(__global float2 *q, __global ushort *output, ushort const maxiter)
    {
      int gid = get_global_id(0);
      float nreal, real = 0;
      float imag = 0;

      output[gid] = 0;

      for(int curiter = 0; curiter < maxiter; curiter++) {
        nreal = real*real - imag*imag + q[gid].x;
        imag = 2* real*imag + q[gid].y;
        real = nreal;

        if (real*real + imag*imag > 4.0f) {
          output[gid] = curiter;
        }
      }
    }
    """).build()

    prg.mandelbrot(queue, output.shape, None, q_opencl, output_opencl, np.uint16(maxiter))

    cl.enqueue_copy(queue, output, output_opencl).wait()

    return output

if __name__ == '__main__':

    class Mandelbrot(object):
        def __init__(self):
            self.w = 3840
            self.h = 2160
            self.fname="mandelbrot.png"
            self.save_image()

        def render(self, x1, x2, y1, y2, maxiter=32):
            xx = np.arange(x1, x2, (x2-x1)/self.w)
            yy = np.arange(y2, y1, (y1-y2)/self.h) * 1j
            q = np.ravel(xx+yy[:, np.newaxis]).astype(np.complex)

            start_main = time.time()
            output = calc_fractal_numpy(q, maxiter)
            end_main = time.time()

            secs = end_main - start_main
            print("Main took", secs)

            self.mandel = (output.reshape((self.h, self.w)) /
                           float(output.max()) * 255.).astype(np.uint8)

        def save_image(self):
            # you can experiment with these x and y ranges
            self.render(-2.13, 2.13, -1.3, 1.3)
            r = self.mandel.astype(np.uint8)
            g = np.zeros_like(self.mandel).astype(np.uint8)
            b = np.zeros_like(self.mandel).astype(np.uint8)
            cv2.imwrite(self.fname, cv2.merge((b, g, r)))
            # self.im = Image.fromarray(self.mandel)
            # self.im.putpalette([i for rgb in ((j, 0, 0) for j in range(255)) for i in rgb])

    # test the class
    test = Mandelbrot()
