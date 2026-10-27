import subprocess, os, json, numpy as np
from PIL import Image
D=r"D:\3DWord\itseez3d\ai-lab\gaussian-splatting\vulkan-rs"
PLY=r"D:\3DWord\itseez3d\ai-lab\ml-sharp\output\photo.ply"
OUT=os.path.join(D,"multiview"); os.makedirs(OUT,exist_ok=True)
exe=os.path.join(D,"target","release","vulkan-splat.exe")
yaws=[-0.6,-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4,0.5,0.6]
# scene centroid from PLY header parse
raw=open(PLY,'rb').read(); hs=raw.index(b'end_header\n')+11; n=1179648
f=np.frombuffer(raw[hs:hs+n*14*4],dtype='<f4').reshape(n,14)
ctr=f[:,:3].mean(axis=0).tolist()
intr_scaled={"fx":1629.2708*832/2048,"fy":1629.2708*480/1152,"cx":1024*832/2048,"cy":576*480/1152,"width":832,"height":480}
cams=[]
for i,yaw in enumerate(yaws):
    ppm=os.path.join(OUT,f"{i:03d}.ppm"); png=os.path.join(OUT,f"{i:03d}.png")
    subprocess.run([exe,PLY,str(yaw),ppm],check=True,stdout=subprocess.DEVNULL)
    Image.open(ppm).convert("RGB").save(png); os.remove(ppm)
    c,s=np.cos(yaw),np.sin(yaw)
    R=np.array([[c,0,-s],[0,1,0],[s,0,c]],np.float32)
    w2c=np.eye(4,dtype=np.float32); w2c[:3,:3]=R; w2c[:3,3]=-R@np.array(ctr,np.float32)
    cams.append({"file":f"{i:03d}.png","yaw":yaw,"w2c":w2c.tolist()})
json.dump({"intrinsics":intr_scaled,"center":ctr,"cameras":cams},open(os.path.join(OUT,"cameras.json"),"w"),indent=1)
print("views",len(cams),OUT)

