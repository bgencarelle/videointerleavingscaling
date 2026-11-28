"""Literal per-pixel transcription of libretro's super-xbr pass0-2 shaders
(Hyllian, MIT), with GLSL semantics: texture coordinates as the vertex
shaders compute them (including the x1.0001), nearest sampling, clamp to
edge. Slow; for tests on tiny images only."""
import math
import numpy as np

Y = (0.2126, 0.7152, 0.0722)
def lum(c): return Y[0]*c[0]+Y[1]*c[1]+Y[2]*c[2]
def df(a,b): return abs(a-b)
def d_wd(wp1,wp2,wp3,wp4,wp5,wp6,b0,b1,c0,c1,c2,d0,d1,d2,d3,e1,e2,e3,f2,f3):
    return (wp1*(df(c1,c2)+df(c1,c0)+df(e2,e1)+df(e2,e3))+wp2*(df(d2,d3)+df(d0,d1))+wp3*(df(d1,d3)+df(d0,d2))+wp4*df(d1,d2)+wp5*(df(c0,c2)+df(e1,e3))+wp6*(df(b0,b1)+df(f2,f3)))
def hv_wd(wp1,wp2,wp3,wp4,wp5,wp6,i1,i2,i3,i4,e1,e2,e3,e4):
    return (wp4*(df(i1,i2)+df(i3,i4))+wp1*(df(i1,e1)+df(i2,e2)+df(i3,e3)+df(i4,e4))+wp3*(df(i1,e2)+df(i3,e4)+df(e1,i2)+df(e3,i4)))
def tex(t, u, v):
    h, w = t.shape[:2]
    x = min(max(int(math.floor(u*w)),0),w-1); y = min(max(int(math.floor(v*h)),0),h-1)
    return t[y, x]
def smooth(e0,e1,x):
    t=min(max((x-e0)/(e1-e0),0.0),1.0); return t*t*(3-2*t)
def body(S, wp, weight, strength):
    (P0,p0),(P1,p1),(P2,p2),(P3,p3)=S["P0"],S["P1"],S["P2"],S["P3"]
    (B,b),(C,c),(H5,h5),(I5,i5)=S["B"],S["C"],S["H5"],S["I5"]
    (D,d),(F4,f4),(G,g),(I4,i4)=S["D"],S["F4"],S["G"],S["I4"]
    (E,e),(F,f),(H,h),(I,i)=S["E"],S["F"],S["H"],S["I"]
    weight1=weight*1.29633/10.0; weight2=weight*1.75068/10.0/2.0
    d_edge=(d_wd(*wp,d,b,g,e,c,p2,h,f,p1,h5,i,f4,i5,i4)-d_wd(*wp,c,f4,b,f,i4,p0,e,i,p3,d,h,i5,g,h5))
    hv_edge=(hv_wd(*wp,f,i,e,h,c,i5,b,h5)-hv_wd(*wp,e,f,h,i,d,f4,g,i4))
    es=smooth(0.0,strength+0.000001,abs(d_edge))
    w1=(-weight1,weight1+0.5,weight1+0.5,-weight1); w2=(-weight2,weight2+0.25,weight2+0.25,-weight2)
    def mm(w,a,b_,c_,d_): return a*w[0]+b_*w[1]+c_*w[2]+d_*w[3]
    c3=mm(w2,D+G,E+H,F+I,F4+I4); c4=mm(w2,C+B,F+E,I+H,I5+H5)
    c1=mm(w1,P2,H,F,P1); c2=mm(w1,P0,E,I,P3)
    a=c2 if d_edge>=0 else c1; bb=c4 if hv_edge>=0 else c3
    col=a*(es)+bb*(1-es)  # mix(a,bb,1-es)
    lo=np.minimum(np.minimum(E,F),np.minimum(H,I)); hi=np.maximum(np.maximum(E,F),np.maximum(H,I))
    return np.clip(col,lo,hi)
def rgba(img):
    return np.concatenate([img, (Y[0]*img[...,0:1]+Y[1]*img[...,1:2]+Y[2]*img[...,2:3])], -1)
def get(t,u,v):
    q=tex(t,u,v); return (q[:3], q[3])
def ref(img, strength=3.0):
    h,w,_=img.shape
    src=rgba(img)
    p0=np.zeros_like(src)
    dx,dy=1.0/w,1.0/h
    for y in range(h):
        for x in range(w):
            u,v=(x+0.5)/w*1.0001,(y+0.5)/h*1.0001
            S={}
            off={"P0":(-1,-1),"P1":(2,-1),"P2":(-1,2),"P3":(2,2),"B":(0,-1),"C":(1,-1),"H5":(0,2),"I5":(1,2),
                 "D":(-1,0),"F4":(2,0),"G":(-1,1),"I4":(2,1),"E":(0,0),"F":(1,0),"H":(0,1),"I":(1,1)}
            for k,(ox,oy) in off.items(): S[k]=get(src,u+ox*dx,v+oy*dy)
            col=body(S,(2.0,1.0,-1.0,4.0,-1.0,1.0),0.0,strength)
            p0[y,x,:3]=col; p0[y,x,3]=lum(col)
    W,H=2*w,2*h
    p1=np.zeros((H,W,4))
    for Yy in range(H):
        for X in range(W):
            u,v=(X+0.5)/W*1.0001,(Yy+0.5)/H*1.0001
            fpx=(u*w)%1.0; fpy=(v*h)%1.0
            dirx,diry=fpx-0.5,fpy-0.5
            if dirx*diry>0:
                p1[Yy,X]= tex(p0,u,v) if dirx>=0 else tex(src,u,v)
                continue
            g1=(0.5/w,0.0) if fpx>0.5 else (0.0,0.5/h)
            g2=(0.0,0.5/h) if fpx>0.5 else (0.5/w,0.0)
            def at(t,k,m): return get(t,u+k*g1[0]+m*g2[0],v+k*g1[1]+m*g2[1])
            S={"P0":at(src,-3,0),"P1":at(p0,0,-3),"P2":at(p0,0,3),"P3":at(src,3,0),
               "B":at(p0,-2,-1),"C":at(src,-1,-2),"D":at(p0,-2,1),"E":at(src,-1,0),
               "F":at(p0,0,-1),"G":at(src,-1,2),"H":at(p0,0,1),"I":at(src,1,0),
               "F4":at(src,1,-2),"I4":at(p0,2,-1),"H5":at(src,1,2),"I5":at(p0,2,1)}
            col=body(S,(1.0,0,0,0,0,0),1.0,strength)
            p1[Yy,X,:3]=col; p1[Yy,X,3]=lum(col)
    out=np.zeros((H,W,3))
    dx,dy=1.0/W,1.0/H
    off2={"P0":(-2,-2),"P1":(1,-2),"P2":(-2,1),"P3":(1,1),"B":(-1,-2),"C":(0,-2),"H5":(-1,1),"I5":(0,1),
          "D":(-2,-1),"F4":(1,-1),"G":(-2,0),"I4":(1,0),"E":(-1,-1),"F":(0,-1),"H":(-1,0),"I":(0,0)}
    for Yy in range(H):
        for X in range(W):
            u,v=(X+0.5)/W*1.0001,(Yy+0.5)/H*1.0001
            S={k:get(p1,u+ox*dx,v+oy*dy) for k,(ox,oy) in off2.items()}
            out[Yy,X]=body(S,(0,0,0,1.0,0,0),1.0,strength)
    return out
