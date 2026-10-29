"""（由獨立審查者撰寫）simulate_side 的擬真版 with: foot-angle kinematics, heel rise in terminal stance (toe pivot),
foot-flat after HS, non-zero heel arrival speed + short post-contact heel slide."""
import sys, numpy as np

from gaitlab import pose as P
from tests.synth_side import SideCamera, _knee, _flex

def herm(ph, m0, m1):
    h00=2*ph**3-3*ph**2+1; h10=ph**3-2*ph**2+ph; h01=-2*ph**3+3*ph**2; h11=ph**3-ph**2
    return h10*m0+h01+h11*m1

def simulate_side_real(speed=1.0, cadence=110.0, stance_frac=0.62, step_width=0.12, fps=60.0, px_noise=2.0,
        seed=0, y_start=9.5, y_end=0.5, marker_y=(4.0,6.0), lane_x=0.5, direction=-1, swing_lift=0.12,
        hr_frac=0.35, hr_deg=40.0, hs_deg=15.0, ff_frac=0.12, v_contact=0.5, slide_s=0.03, Lf=0.2):
    rng=np.random.default_rng(seed)
    step_t=60.0/cadence; step_len=speed*step_t; stride_t=2*step_t; st_dur=stance_frac*stride_t
    if direction>0: y_start,y_end=y_end,y_start
    n_steps=int(abs(y_start-y_end)/step_len)+2
    places,sides,times=[],[],[]
    for k in range(n_steps):
        s="L" if k%2==0 else "R"
        lat=step_width/2*(1 if s=="L" else -1)*(-np.sign(direction))
        places.append([lane_x+lat, y_start+direction*k*step_len]); sides.append(s); times.append(k*step_t)
    places,times=np.array(places),np.array(times)
    T=int(times[-1]*fps); t=np.arange(T)/fps
    heel={s:np.zeros((T,3)) for s in "LR"}; toe={s:np.zeros((T,3)) for s in "LR"}
    slide=v_contact*slide_s/2
    def stance_pose(k, tau):  # tau = time since HS
        y0=places[k][1]
        if tau<slide_s: hy=y0-direction*slide+direction*(v_contact*tau-v_contact*tau**2/(2*slide_s))
        else: hy=y0
        if tau<ff_frac*st_dur:
            a=hs_deg*(1-tau/(ff_frac*st_dur)); hz=0.03
            th=np.radians(a); return hy,hz,hy+direction*Lf*np.cos(th),0.03+Lf*np.sin(th)
        hr0=(1-hr_frac)*st_dur
        if tau<hr0: return hy,0.03,hy+direction*Lf,0.03
        u=(tau-hr0)/(hr_frac*st_dur); a=-hr_deg*(1-np.cos(np.pi*u))/2; th=np.radians(a)
        ty=y0+direction*Lf
        return ty-direction*Lf*np.cos(th), 0.03-Lf*np.sin(th), ty, 0.03
    for s in "LR":
        idx=[k for k in range(n_steps) if sides[k]==s]
        for i,ti in enumerate(t):
            prev=[k for k in idx if times[k]<=ti]; k0=prev[-1] if prev else idx[0]
            nxt=[k for k in idx if k>k0]
            tau=ti-times[k0]
            if (not prev) or tau<st_dur or not nxt:
                hy,hz,ty,tz=stance_pose(k0, max(tau,0) if prev else 0.0)
            else:
                hy0,hz0,_,_=stance_pose(k0, st_dur)
                dt_eps=1e-3; hya,_,_,_=stance_pose(k0, st_dur-dt_eps); v0=(hy0-hya)/dt_eps
                k1=nxt[0]; sw=times[k1]-times[k0]-st_dur
                yend=places[k1][1]-direction*slide
                D=yend-hy0; ph=(tau-st_dur)/sw
                m0=v0*sw/D; m1=direction*v_contact*sw/D
                hy=hy0+D*herm(ph,m0,m1)
                hz=hz0*(1-ph)+0.03*ph+swing_lift*np.sin(np.pi*ph)
                a=-hr_deg+(hr_deg+hs_deg)*(1-np.cos(np.pi*ph))/2; th=np.radians(a)
                ty=hy+direction*Lf*np.cos(th); tz=hz+Lf*np.sin(th)
            heel[s][i]=[places[k0][0] if True else 0,hy,hz]; toe[s][i]=[places[k0][0],ty,tz]
    yc=y_start+direction*(speed*t+step_len/2)
    fwd=np.array([0,float(np.sign(direction)),0])
    hip={"L":np.stack([np.full(T,lane_x+0.16*(-np.sign(direction))),yc,np.full(T,0.92)],-1),
         "R":np.stack([np.full(T,lane_x-0.16*(-np.sign(direction))),yc,np.full(T,0.92)],-1)}
    ankle={s:heel[s]+np.array([0,direction*0.05,0.07]) for s in "LR"}
    knee={s:np.array([_knee(hip[s][i],ankle[s][i],fwd) for i in range(T)]) for s in "LR"}
    cam=SideCamera(along=float(np.mean(marker_y)),lane_x=lane_x)
    pts={P.L_HEEL:heel["L"],P.R_HEEL:heel["R"],P.L_TOE:toe["L"],P.R_TOE:toe["R"],P.L_ANKLE:ankle["L"],
         P.R_ANKLE:ankle["R"],P.L_KNEE:knee["L"],P.R_KNEE:knee["R"],P.L_HIP:hip["L"],P.R_HIP:hip["R"]}
    xy=np.zeros((T,P.N_LANDMARKS,2)); mid=(hip["L"]+hip["R"])/2+[0,0,0.5]
    for k in range(P.N_LANDMARKS): xy[:,k]=cam.project(pts.get(k,mid))
    xy+=rng.normal(0,px_noise,xy.shape)
    lm={"xy":xy,"vis":np.ones((T,P.N_LANDMARKS)),"t":t,"fps":fps,"size":(cam.w,cam.h)}
    mk=cam.project(np.array([[lane_x,marker_y[0],0],[lane_x,marker_y[1],0]]))+rng.normal(0,1.0,(2,2))
    truth={"stance_pct":stance_frac*100,"double_support_pct":(2*stance_frac-1)*100,"cadence_steps_min":cadence,
           "step_length_cm":step_len*100,"gait_speed_cm_s":speed*100,"knee_peak_swing_deg":np.nan}
    return lm,mk,truth,(heel,toe,t)
