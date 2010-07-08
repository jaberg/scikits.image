from scikits.lazy import register_transform, transform_db

from scikits.image.color import colorconv

_enabled = True
def enable():
    global _enabled
    _enabled=True
def disable():
    global _enabled
    _enabled=False

import pyopencl as cl
try:
    _cpu_context = cl.Context(dev_type=cl.device_type.CPU)
    _cpu_queue = cl.CommandQueue(_cpu_context)
except cl.LogicError:
    _cpu_context = None
    _cpu_queue = None


@register_transform(transform_db['begin_backend'].position + 0.1, 'default')
def replace_colorconvert_with_opencl(closure, **kwargs):
    if not _enabled or _cpu_context is None: return

    replacements = []
    for expr in closure.expr_iter_with_Impl(colorconv.ColorConvert3x3):
        m3x3, img = expr.inputs
        new_impl = ColorConvert3x3_OpenCL.build_for_types(m3x3.type.dtype, img.type.dtype)
        new_out = new_impl(m3x3, img)
        replacements.append((expr.outputs[0], new_out))
    for swap in replacements:
        closure.replace_symbol(*swap)


ctype_from_dtype = {
    numpy.dtype('float32'):'float',
    numpy.dtype('float64'):'double',
    }

class ColorConvert3x3_OpenCL(colorconv.ColorConvert3x3):
    
    @classmethod
    def build_for_types(cls, m3x3_dtype, img_dtype):

        i_ctype = ctype_from_dtype[img_dtype]
        m_ctype = ctype_from_dtype[m3x3_dtype]
        o_ctype = ctype_from_dtype[img_dtype]

        prg = cl.Program(_cpu_context, """
            __kernel void elemwise(
                const int N,
                __global const %(i_ctype)s *img,
                __global const %(m_ctype)s *m3x3,
                __global %(o_ctype)s *out
                )
            {
                const %(m_ctype)s m00 = m3x3[0];
                const %(m_ctype)s m01 = m3x3[1];
                const %(m_ctype)s m02 = m3x3[2];
                const %(m_ctype)s m10 = m3x3[3];
                const %(m_ctype)s m11 = m3x3[4];
                const %(m_ctype)s m12 = m3x3[5];
                const %(m_ctype)s m20 = m3x3[6];
                const %(m_ctype)s m21 = m3x3[7];
                const %(m_ctype)s m22 = m3x3[8];
                int gid = get_global_id(0);
                for (int i = N*gid; i < N*gid+N; ++i)
                {
                    %(i_ctype)s i0 = img[i*3+0];
                    %(i_ctype)s i1 = img[i*3+1];
                    %(i_ctype)s i2 = img[i*3+2];
                    %(o_ctype)s o0 = i0*m00 + i1*m01 + i2*m02;
                    %(o_ctype)s o1 = i0*m10 + i1*m11 + i2*m12;
                    %(o_ctype)s o2 = i0*m20 + i1*m21 + i2*m22;
                    out[i*3+0] = o0;
                    out[i*3+1] = o1;
                    out[i*3+2] = o2;
                }
            }
            """ % locals()).build()

        def rval(m3x3, img):
            nthreads=1 
            n_pixels = img.shape[0]*img.shape[1]
            if n_pixels > 1000:
                #TODO: heuristic to choose how many threads
                n_threads = 4
            
            z = numpy.empty_like(a) #empty_like is faster, but can hide errors
            a_buf = cl.Buffer(_cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=a)
            m_buf = cl.Buffer(_cpu_context, cl.mem_flags.READ_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=m3x3)
            z_buf = cl.Buffer(_cpu_context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.USE_HOST_PTR, hostbuf=z)
            rval = prg.elemwise(_cpu_queue, (nthreads,), None, 
                    numpy.int64(n_pixels/nthreads), a_buf, m_buf, z_buf)
            rval.wait()  #not good if there are several OpenCL commands to do in sequence
            return z
        
        return cls(fn=rval, name=cls.__name__)


