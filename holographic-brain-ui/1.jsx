import { useState, useRef, useMemo, useEffect, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Torus } from "@react-three/drei";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { gunzipSync, strFromU8 } from "fflate";
import anatomyLabHtml from "./HoloAnatomyLab.html?raw";

// 
// SHARED SHADERS
// 
const HOLO_VERT = `varying vec2 vUv;varying vec3 vNormal;varying vec3 vPosition;
void main(){vUv=uv;vNormal=normalize(normalMatrix*normal);vPosition=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);}`;
const HOLO_FRAG = `uniform float time;uniform vec3 color;uniform float opacity;
varying vec2 vUv;varying vec3 vNormal;varying vec3 vPosition;
void main(){
  float scan=sin(vUv.y*80.0+time*3.5)*0.06+0.94;
  float fresnel=pow(1.0-abs(dot(vNormal,vec3(0.0,0.0,1.0))),2.6);
  float pulse=sin(time*1.8+vPosition.y*2.5)*0.04+0.96;
  float fiber=sin(vPosition.x*16.0+vPosition.y*10.0)*0.012+sin(vPosition.y*20.0+vPosition.z*7.0)*0.008;
  float alpha=(0.22+fresnel*0.5+fiber)*scan*pulse*opacity;
  gl_FragColor=vec4(color+vec3(fresnel*0.22),alpha);
}`;
const VESSEL_FRAG = `uniform float time;uniform vec3 color;uniform float opacity;
varying vec2 vUv;varying vec3 vNormal;varying vec3 vPosition;
void main(){
  float scan=sin(vUv.y*80.0+time*4.0)*0.05+0.95;
  float fresnel=pow(1.0-abs(dot(vNormal,vec3(0.0,0.0,1.0))),2.0);
  float flow=sin(vUv.y*6.0-time*8.0)*0.07+0.93;
  float alpha=(0.42+fresnel*0.44)*scan*flow*opacity;
  gl_FragColor=vec4(color+vec3(fresnel*0.15),alpha);
}`;
const BEAT_FRAG = `uniform float time;uniform vec3 color;uniform float opacity;
varying vec2 vUv;varying vec3 vNormal;varying vec3 vPosition;
void main(){
  float scan=sin(vUv.y*120.0+time*3.0)*0.04+0.96;
  float fresnel=pow(1.0-abs(dot(vNormal,vec3(0.0,0.0,1.0))),2.8);
  float beat=sin(time*5.5)*0.5+0.5; float bp=beat*beat;
  float pulse=0.92+bp*0.08;
  float fiber=sin(vPosition.x*18.0+vPosition.y*12.0)*0.015+sin(vPosition.y*22.0+vPosition.z*8.0)*0.01;
  float alpha=(0.28+fresnel*0.52+fiber)*scan*opacity;
  vec3 col=color*(pulse+fiber*2.0)+vec3(fresnel*0.2);
  gl_FragColor=vec4(col,alpha);
}`;

function HoloMesh({ geo, color, opacity=1.0, frag=HOLO_FRAG, wire=false }) {
  const r = useRef();
  const u = useMemo(() => ({ time:{value:0}, color:{value:new THREE.Color(color)}, opacity:{value:opacity} }), [color, opacity]);
  useFrame(({clock}) => { if(r.current) r.current.uniforms.time.value = clock.elapsedTime; });
  if(wire) return <mesh geometry={geo}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35} transparent opacity={opacity*0.3} wireframe/></mesh>;
  return <mesh geometry={geo}><shaderMaterial ref={r} vertexShader={HOLO_VERT} fragmentShader={frag} uniforms={u} transparent side={THREE.DoubleSide} depthWrite={false}/></mesh>;
}

const mkTube = (pts, r=0.04, s=6) => new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(...p))), 32, r, s, false);
const shiftBlue = (hex) => { const c=new THREE.Color(hex); c.r*=0.4; c.g*=0.6; c.b=Math.min(1,c.b*1.4+0.2); return `#${c.getHexString()}`; };
const brightenC = (hex,f) => { const c=new THREE.Color(hex); c.r=Math.min(1,c.r*f+0.15); c.g=Math.min(1,c.g*f+0.05); c.b=Math.min(1,c.b*f+0.05); return `#${c.getHexString()}`; };
const extractEmbeddedConst = (html, constName) => {
  const rx = new RegExp(`const\\s+${constName}\\s*=\\s*"([\\s\\S]*?)";`);
  const m = html.match(rx);
  return m ? m[1] : null;
};
const HUMAN_ATLAS_B64 = extractEmbeddedConst(anatomyLabHtml, "HUMAN_B64");
let humanAtlasModelPromise = null;

async function b64GzipToText(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return strFromU8(gunzipSync(bytes));
}

function parseOBJText(text) {
  const vp = [];
  const vn = [];
  const verts = [];
  for (const line of text.split("\n")) {
    const p = line.trim().split(/\s+/);
    if (p[0] === "v") vp.push(+p[1], +p[2], +p[3]);
    else if (p[0] === "vn") vn.push(+p[1], +p[2], +p[3]);
    else if (p[0] === "f") {
      const face = [];
      for (let i = 1; i < p.length; i++) {
        const idx = p[i].split("/");
        face.push({ v: parseInt(idx[0], 10) - 1, vn: idx[2] ? parseInt(idx[2], 10) - 1 : -1 });
      }
      for (let i = 1; i < face.length - 1; i++) verts.push(face[0], face[i], face[i + 1]);
    }
  }
  const pa = new Float32Array(verts.length * 3);
  const na = new Float32Array(verts.length * 3);
  for (let i = 0; i < verts.length; i++) {
    const { v, vn: vni } = verts[i];
    pa[i * 3] = vp[v * 3];
    pa[i * 3 + 1] = vp[v * 3 + 1];
    pa[i * 3 + 2] = vp[v * 3 + 2];
    if (vni >= 0) {
      na[i * 3] = vn[vni * 3];
      na[i * 3 + 1] = vn[vni * 3 + 1];
      na[i * 3 + 2] = vn[vni * 3 + 2];
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(pa, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(na, 3));
  if (vn.length === 0) g.computeVertexNormals();
  return g;
}

function centerScaleGeo(g, targetHeight) {
  g.computeBoundingBox();
  const box = g.boundingBox;
  const ctr = new THREE.Vector3();
  const sz = new THREE.Vector3();
  box.getCenter(ctr);
  box.getSize(sz);
  g.translate(-ctr.x, -ctr.y, -ctr.z);
  return targetHeight / sz.y;
}

function loadHumanAtlasModel() {
  if (humanAtlasModelPromise) return humanAtlasModelPromise;
  humanAtlasModelPromise = (async () => {
    if (!HUMAN_ATLAS_B64) throw new Error("HUMAN_B64 not found in HoloAnatomyLab.html");
    const humanText = await b64GzipToText(HUMAN_ATLAS_B64);
    const geometry = parseOBJText(humanText);
    const scale = centerScaleGeo(geometry, 3.2);
    geometry.computeVertexNormals();
    return { geometry, scale };
  })();
  return humanAtlasModelPromise;
}

// 
// GEOMETRY BUILDERS
// 
function buildBrain(rat=false) {
  if (rat) {
    const cerebrum = new THREE.SphereGeometry(0.95, 112, 112).toNonIndexed();
    const cp = cerebrum.attributes.position;
    for (let i = 0; i < cp.count; i++) {
      let x = cp.getX(i), y = cp.getY(i), z = cp.getZ(i);
      x *= 0.66; y *= 0.52; z *= 1.56;
      const dorsalArch = 1 + Math.exp(-((z - 0.1) ** 2) / 1.2) * 0.08;
      y *= dorsalArch;
      const frontalPole = Math.exp(-((z - 1.2) ** 2) / 0.11) * 0.16;
      z += frontalPole;
      const temporalInset = Math.exp(-((Math.abs(x) - 0.38) ** 2) / 0.05) * Math.exp(-((y + 0.04) ** 2) / 0.1) * 0.1;
      x *= 1 + temporalInset;
      const fissure = Math.exp(-(x * x) / 0.006) * Math.exp(-((y - 0.06) ** 2) / 0.12) * 0.085;
      x -= Math.sign(x || 1) * fissure;
      const micro = (Math.sin(z * 10 + y * 7) + Math.cos(x * 12 - z * 5)) * 0.0045;
      const s = 1 + micro;
      cp.setXYZ(i, x * s, y * s, z * s);
    }

    const bulbL = new THREE.SphereGeometry(0.2, 48, 48).toNonIndexed();
    bulbL.scale(0.62, 0.48, 1.08);
    bulbL.translate(0.12, 0.02, 1.38);
    const bulbR = bulbL.clone();
    bulbR.translate(-0.24, 0, 0);

    const cerebellum = new THREE.SphereGeometry(0.34, 64, 64).toNonIndexed();
    const rb = cerebellum.attributes.position;
    for (let i = 0; i < rb.count; i++) {
      let x = rb.getX(i), y = rb.getY(i), z = rb.getZ(i);
      x *= 0.86; y *= 0.58; z *= 0.8;
      const folia = Math.sin(z * 24 + y * 8) * 0.014 + Math.cos(x * 17) * 0.01;
      const s = 1 + folia;
      rb.setXYZ(i, x * s, y * s, z * s);
    }
    cerebellum.translate(0, -0.24, -0.98);

    const stem = new THREE.CapsuleGeometry(0.08, 0.34, 8, 16).toNonIndexed();
    stem.rotateX(0.34);
    stem.translate(0, -0.44, -0.76);

    const out = mergeGeometries([cerebrum, bulbL, bulbR, cerebellum, stem], false);
    out.computeVertexNormals();
    return out;
  }

  const cerebrum = new THREE.SphereGeometry(1.06, 128, 128).toNonIndexed();
  const cp = cerebrum.attributes.position;
  for (let i = 0; i < cp.count; i++) {
    let x = cp.getX(i), y = cp.getY(i), z = cp.getZ(i);
    x *= 1.16; y *= 0.93; z *= 1.04;
    const frontalBulge = Math.exp(-((z - 0.88) ** 2) / 0.15) * 0.09;
    const occipitalBulge = Math.exp(-((z + 0.88) ** 2) / 0.12) * 0.11;
    z += frontalBulge - occipitalBulge * 0.2;
    const temporalBulge = Math.exp(-((Math.abs(x) - 0.78) ** 2) / 0.08) * Math.exp(-((y + 0.05) ** 2) / 0.3) * 0.12;
    x *= 1 + temporalBulge;
    const fissure = Math.exp(-(x * x) / 0.012) * Math.exp(-((y - 0.22) ** 2) / 0.45) * 0.15;
    x -= Math.sign(x || 1) * fissure;
    const gyri = Math.sin(z * 18 + y * 11) * 0.017 + Math.sin(x * 21 - z * 8) * 0.014 + Math.cos(y * 17 + x * 10) * 0.012;
    const s = 1 + gyri;
    cp.setXYZ(i, x * s, y * s, z * s);
  }

  const cerebellum = new THREE.SphereGeometry(0.44, 72, 72).toNonIndexed();
  const rb = cerebellum.attributes.position;
  for (let i = 0; i < rb.count; i++) {
    let x = rb.getX(i), y = rb.getY(i), z = rb.getZ(i);
    x *= 0.92; y *= 0.62; z *= 0.74;
    const folia = Math.sin(z * 28 + y * 9) * 0.02 + Math.cos(x * 18 - z * 7) * 0.012;
    const s = 1 + folia;
    rb.setXYZ(i, x * s, y * s, z * s);
  }
  cerebellum.translate(0, -0.56, -0.69);

  const stem = new THREE.CapsuleGeometry(0.11, 0.34, 8, 18).toNonIndexed();
  stem.rotateX(0.3);
  stem.translate(0, -0.88, -0.42);

  const out = mergeGeometries([cerebrum, cerebellum, stem], false);
  out.computeVertexNormals();
  return out;
}

function buildHeart(rat=false) {
  const g = new THREE.SphereGeometry(1, rat ? 208 : 272, rat ? 208 : 272).toNonIndexed();
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    const rr = Math.sqrt(x * x + y * y + z * z) + 0.0001;
    const phi = Math.acos(y / rr);
    const ap = phi / Math.PI; // 0 base, 1 apex
    const az = Math.atan2(z, x);

    if (rat) {
      x *= 0.76;
      y *= 1.05;
      z *= 0.64;

      const cone = 1 - ap * 0.56;
      const apex = Math.pow(ap, 1.95);
      x += -0.18 * apex;
      y -= 0.16 * apex;
      z += 0.08 * apex;

      const rv = Math.exp(-((az - 0.1) ** 2) / 0.42) * Math.exp(-((ap - 0.58) ** 2) / 0.08) * 0.22;
      const lv = Math.exp(-((az - Math.PI + 0.2) ** 2) / 0.55) * Math.exp(-((ap - 0.63) ** 2) / 0.1) * 0.18;
      const atria = Math.exp(-((ap - 0.18) ** 2) / 0.03) * (0.14 + 0.08 * Math.cos(az * 2.0));
      const baseNotch = -Math.exp(-((ap - 0.09) ** 2) / 0.008) * 0.05;
      const coronary = -Math.exp(-((ap - 0.3) ** 2) / 0.012) * 0.06;
      const ivSulcus = -Math.exp(-((az - 0.22) ** 2) / 0.045) * Math.exp(-((ap - 0.58) ** 2) / 0.12) * 0.09;
      const trabec = Math.sin(phi * 16 + az * 7) * 0.01 + Math.sin(phi * 22 - az * 5) * 0.007;
      const posterior = Math.exp(-((az + Math.PI / 2) ** 2) / 0.4) * Math.exp(-((ap - 0.55) ** 2) / 0.16) * 0.06;

      const sc = cone + rv + lv + atria + baseNotch + coronary + ivSulcus + posterior + trabec;
      p.setXYZ(i, x * sc, y * sc, z * sc);
      continue;
    }

    x *= 0.9;
    y *= 0.98;
    z *= 0.78;

    const cone = 1 - ap * 0.46;
    const apex = Math.pow(ap, 1.85);
    x += -0.24 * apex;
    y -= 0.1 * apex;
    z += 0.16 * apex;

    const rv = Math.exp(-((az - 0.0) ** 2) / 0.38) * Math.exp(-((ap - 0.56) ** 2) / 0.09) * 0.26;
    const lv = Math.exp(-((az - Math.PI + 0.24) ** 2) / 0.58) * Math.exp(-((ap - 0.62) ** 2) / 0.1) * 0.21;
    const atria = Math.exp(-((ap - 0.17) ** 2) / 0.03) * (0.16 + 0.08 * Math.cos(az * 2.0));
    const appendages = Math.exp(-((ap - 0.13) ** 2) / 0.014) * (0.08 + 0.05 * Math.sin(az * 3.2));
    const baseNotch = -Math.exp(-((ap - 0.08) ** 2) / 0.008) * 0.06;
    const coronary = -Math.exp(-((ap - 0.3) ** 2) / 0.012) * 0.075;
    const ivSulcus = -Math.exp(-((az - 0.26) ** 2) / 0.05) * Math.exp(-((ap - 0.56) ** 2) / 0.13) * 0.1;
    const posterior = Math.exp(-((az + Math.PI / 2.2) ** 2) / 0.38) * Math.exp(-((ap - 0.58) ** 2) / 0.14) * 0.07;
    const surface = Math.sin(phi * 14 + az * 8) * 0.011 + Math.sin(phi * 21 - az * 6) * 0.008 + Math.cos(phi * 11 + az * 9) * 0.006;

    const sc = cone + rv + lv + atria + appendages + baseNotch + coronary + ivSulcus + posterior + surface;
    p.setXYZ(i, x * sc, y * sc, z * sc);
  }
  g.computeVertexNormals();
  return g;
}

function buildLiver(rat=false) {
  const g = new THREE.SphereGeometry(rat?0.72:1, 96, 96);
  const p = g.attributes.position;
  for(let i=0;i<p.count;i++){
    let x=p.getX(i),y=p.getY(i),z=p.getZ(i);
    if(rat){ const l1=Math.sin(x*2.5+1.2)*0.18,l2=Math.sin(x*2.5-1.2)*0.14; const lf=1+l1*(y>0?1:0.3)+l2; const n=Math.sin(x*8+y*5)*0.032+Math.sin(y*11+z*6)*0.025; p.setXYZ(i,x*lf*(1+n),y*(y<-0.2?1+(y+0.2)*0.5:1)*(1+n)*0.68,z*(1+n)*0.74); }
    else { const rb=x>0?1+x*0.35:1+x*0.12; const fb=y<-0.3?1+(y+0.3)*0.4:1; const af=z>0.3?1-(z-0.3)*0.25:1; const n=Math.sin(x*6+y*4)*0.028+Math.sin(y*9+z*5)*0.022+Math.cos(z*7+x*6)*0.018; const sc=rb*fb*af; p.setXYZ(i,x*sc*(1+n)*1.1,y*(1+n)*0.72,z*(1+n)*0.78); }
  }
  g.computeVertexNormals(); return g;
}

// Body geometry builders
const blobGeo = (sx,sy,sz,segs=48) => { const g=new THREE.SphereGeometry(1,segs,segs); const p=g.attributes.position; for(let i=0;i<p.count;i++) p.setXYZ(i,p.getX(i)*sx,p.getY(i)*sy,p.getZ(i)*sz); g.computeVertexNormals(); return g; };

function buildTorso() {
  const g = new THREE.SphereGeometry(1, 96, 96); const p=g.attributes.position;
  for(let i=0;i<p.count;i++){
    let x=p.getX(i),y=p.getY(i),z=p.getZ(i);
    const yn=(y+1)/2;
    const chest=yn>0.55?1+(yn-0.55)*0.35:1;
    const waist=(yn>0.3&&yn<0.55)?1-(0.55-yn)/0.25*0.12:1;
    const aflt=z>0.3?1-(z-0.3)*0.15:1;
    const rib=yn>0.4?Math.sin(y*18)*0.018:0;
    const n=Math.sin(x*8+y*5)*0.01+Math.cos(z*6+y*7)*0.008;
    const sc=chest*waist*aflt;
    p.setXYZ(i,x*sc*(1+n+rib*0.3),y*(1+n),z*sc*0.82*(1+n));
  }
  g.computeVertexNormals(); return g;
}

function buildHead() {
  const g=new THREE.SphereGeometry(0.42,64,64); const p=g.attributes.position;
  for(let i=0;i<p.count;i++){
    let x=p.getX(i),y=p.getY(i),z=p.getZ(i);
    const r=Math.sqrt(x*x+y*y+z*z)+0.001; const phi=Math.acos(y/r);
    const fp=z>0?1+z*0.22*Math.max(0,1-phi/Math.PI*2):1;
    const occ=z<-0.1&&y<0.1?1+Math.abs(z)*0.1:1;
    const n=Math.sin(x*12+y*8)*0.008+Math.sin(y*15+z*6)*0.006;
    p.setXYZ(i,x*(fp+n)*0.98,y*(1+n),z*fp*occ*(1+n));
  }
  g.computeVertexNormals(); return g;
}

const buildPelvis=()=>{const g=new THREE.SphereGeometry(0.68,48,48);const p=g.attributes.position;for(let i=0;i<p.count;i++){let x=p.getX(i),y=p.getY(i),z=p.getZ(i);const ft=y>0.2?1-(y-0.2)*0.6:1;const w=Math.abs(x)>0.3?1+(Math.abs(x)-0.3)*0.4:1;p.setXYZ(i,x*w*ft,y*0.55,z*0.72*ft);}g.computeVertexNormals();return g;};
const mkTubeBody=(pts,r,s=7)=>new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(p=>new THREE.Vector3(...p))),28,r,s,false);

function buildSpine(){const pts=[];for(let i=0;i<=24;i++){const t=i/24;pts.push([0,1.85-t*3.2,-0.08+Math.sin(t*Math.PI*2.5)*0.04]);}return mkTubeBody(pts,0.028,6);}
function buildLungGeo(side){const g=new THREE.SphereGeometry(0.32,32,32);const p=g.attributes.position;for(let i=0;i<p.count;i++){let x=p.getX(i),y=p.getY(i),z=p.getZ(i);const med=Math.abs(x)<0.15?1-(0.15-Math.abs(x))/0.15*0.5:1;p.setXYZ(i,x*med*0.85,y*1.55,z*0.72);}g.computeVertexNormals();return g;}
function buildBodyHeart(){const g=new THREE.SphereGeometry(0.22,48,48);const p=g.attributes.position;for(let i=0;i<p.count;i++){let x=p.getX(i),y=p.getY(i),z=p.getZ(i);const r2=Math.sqrt(x*x+y*y+z*z)+0.001;const phi=Math.acos(y/r2);const t2=1-(phi/Math.PI)*0.4;const rv=z>0&&x>-0.1?Math.max(0,z)*0.3*Math.sin(phi*Math.PI):0;const n=Math.sin(x*14+y*8)*0.015;p.setXYZ(i,x*t2*(1+n+rv*0.2),y*(t2+n)*0.9,z*t2*(1+n+rv));}g.computeVertexNormals();return g;}
function buildBodyBrain(){const g=new THREE.SphereGeometry(0.36,64,64).toNonIndexed();const p=g.attributes.position;for(let i=0;i<p.count;i++){let x=p.getX(i),y=p.getY(i),z=p.getZ(i);x*=1.12;y*=0.9;z*=1.0;const fiss=Math.exp(-(x*x)/0.014)*Math.exp(-((y-0.08)**2)/0.22)*0.06;x-=Math.sign(x||1)*fiss;const gyri=Math.sin(z*16+y*10)*0.012+Math.cos(x*18-z*6)*0.01;const s=1+gyri;p.setXYZ(i,x*s,y*s,z*s);}g.computeVertexNormals();return g;}
function buildBodyLiver(){const g=new THREE.SphereGeometry(0.2,32,32);const p=g.attributes.position;for(let i=0;i<p.count;i++){let x=p.getX(i),y=p.getY(i),z=p.getZ(i);const rb=x>0?1+x*0.35:1+x*0.12;const n=Math.sin(x*7+y*5)*0.02;p.setXYZ(i,x*rb*(1+n)*1.1,y*(1+n)*0.72,z*(1+n)*0.78);}g.computeVertexNormals();return g;}

// 
// SHARED EFFECTS
// 
function ScanRings({ color, radii=[1.6,1.82,2.04], speed=[0.4,0.3,0.2] }) {
  const refs=[useRef(),useRef(),useRef()];
  useFrame(({clock})=>{const t=clock.elapsedTime;refs[0].current&&(refs[0].current.rotation.x=t*speed[0]);refs[1].current&&(refs[1].current.rotation.y=t*speed[1]);refs[2].current&&(refs[2].current.rotation.x=t*speed[2],refs[2].current.rotation.z=t*0.14);});
  return <>{refs.map((r,i)=><Torus key={i} ref={r} args={[radii[i],0.007,8,90]}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.2} transparent opacity={0.3-i*0.07}/></Torus>)}</>;
}
function MRIPlane({ color, range=1.1, spd=0.75 }) {
  const r=useRef();
  useFrame(({clock})=>{if(r.current){r.current.position.y=Math.sin(clock.elapsedTime*spd)*range;r.current.material.opacity=0.06+Math.abs(Math.sin(clock.elapsedTime*spd))*0.05;}});
  return <mesh ref={r} rotation={[-Math.PI/2,0,0]}><planeGeometry args={[4,4]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={2} transparent opacity={0.07} side={THREE.DoubleSide}/></mesh>;
}
function Particles({ color, count=1200, spread=1.8, beat=false }) {
  const r=useRef();
  const {pos,spd}=useMemo(()=>{const pos=new Float32Array(count*3),spd=new Float32Array(count);for(let i=0;i<count;i++){const th=Math.random()*Math.PI*2,ph=Math.acos(2*Math.random()-1),rr=spread*0.6+Math.random()*spread*0.5;pos[i*3]=rr*Math.sin(ph)*Math.cos(th);pos[i*3+1]=rr*Math.sin(ph)*Math.sin(th);pos[i*3+2]=rr*Math.cos(ph);spd[i]=0.2+Math.random()*0.6;}return{pos,spd};},[count,spread]);
  useFrame(({clock})=>{if(!r.current)return;const a=r.current.geometry.attributes.position.array,t=clock.elapsedTime;const bm=beat?Math.pow(Math.max(0,Math.sin(t*5.5)),2)*0.002:0;for(let i=0;i<count;i++){const sp=(0.0006+bm)*spd[i];const ag=t*spd[i]*2+i;a[i*3]+=Math.sin(ag*1.3)*sp;a[i*3+1]+=Math.cos(ag*0.9)*sp*0.7;a[i*3+2]+=Math.sin(ag*1.1+1)*sp*0.8;const d=Math.sqrt(a[i*3]**2+a[i*3+1]**2+a[i*3+2]**2);if(d>spread*1.6){a[i*3]*=0.3;a[i*3+1]*=0.3;a[i*3+2]*=0.3;}}r.current.geometry.attributes.position.needsUpdate=true;});
  return <points ref={r}><bufferGeometry><bufferAttribute attach="attributes-position" count={count} array={pos} itemSize={3}/></bufferGeometry><pointsMaterial size={0.014} color={color} transparent opacity={0.6} sizeAttenuation/></points>;
}

function ECGLine({ color }) {
  const r=useRef();
  useEffect(()=>{
    const cv=r.current;if(!cv)return;const ctx=cv.getContext("2d");let id,t=0;
    const ecg=x=>{const c=x%1;if(c<0.08)return Math.sin(c/0.08*Math.PI)*0.15;if(c<0.18)return 0;if(c<0.20)return-0.12;if(c<0.24)return Math.sin((c-0.2)/0.04*Math.PI)*1.0;if(c<0.27)return-0.18;if(c<0.38)return-0.05;if(c<0.52)return Math.sin((c-0.38)/0.14*Math.PI)*0.28;return 0;};
    const draw=()=>{ctx.clearRect(0,0,cv.width,cv.height);ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.shadowBlur=6;ctx.shadowColor=color;ctx.globalAlpha=0.9;ctx.beginPath();for(let px=0;px<cv.width;px++){const y=cv.height/2-ecg((px/cv.width)*2.5+t)*cv.height/2*0.82;px===0?ctx.moveTo(px,y):ctx.lineTo(px,y);}ctx.stroke();ctx.fillStyle="rgba(3,5,10,0.12)";ctx.fillRect(0,0,cv.width,cv.height);t+=0.012;id=requestAnimationFrame(draw);};
    draw();return()=>cancelAnimationFrame(id);
  },[color]);
  return <canvas ref={r} width={280} height={52} style={{display:"block",background:"transparent"}}/>;
}

// 
// ORGAN SCENES
// 
function BrainLabel({ color, label, anchor, offset }) {
  const tip = [anchor[0] + offset[0], anchor[1] + offset[1], anchor[2] + offset[2]];
  const line = useMemo(
    () => new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...anchor), new THREE.Vector3(...tip)]),
    [anchor, tip]
  );
  return <group>
    <line geometry={line}><lineBasicMaterial color={color} transparent opacity={0.66} /></line>
    <mesh position={tip}>
      <boxGeometry args={[0.055, 0.008, 0.055]} />
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.5} transparent opacity={0.55} />
    </mesh>
  </group>;
}

function BrainLobes({ color, isRat }) {
  const structs = isRat
    ? [
        { l:"Olfactory Bulb", p:[0.02,0.02,1.36], o:[0.34,0.05,0.12] },
        { l:"Neocortex", p:[0.18,0.26,0.42], o:[0.42,0.16,0.16] },
        { l:"Cerebellum", p:[0.02,-0.2,-0.95], o:[0.34,-0.08,-0.18] },
        { l:"Brainstem", p:[0.0,-0.46,-0.74], o:[0.34,-0.18,-0.1] },
      ]
    : [
        { l:"Frontal Lobe", p:[0.28,0.2,0.88], o:[0.5,0.2,0.14] },
        { l:"Parietal Lobe", p:[0.34,0.46,0.08], o:[0.56,0.26,0.08] },
        { l:"Cerebellum", p:[0.06,-0.52,-0.66], o:[0.45,-0.2,-0.18] },
        { l:"Brainstem", p:[0.0,-0.86,-0.44], o:[0.4,-0.16,-0.1] },
      ];
  return <>{structs.map((s) => <BrainLabel key={s.l} color={color} label={s.l} anchor={s.p} offset={s.o} />)}</>;
}

function HeartVessels({ color, isRat }) {
  const aGeo=useMemo(()=>mkTube([[0.02,0.58,0.22],[0.03,0.83,0.33],[0.14,1.05,0.34],[0.33,1.16,0.22],[0.5,1.08,0.03],[0.58,0.92,-0.16],[0.6,0.68,-0.26]],0.092,12),[]);
  const pGeo=useMemo(()=>mkTube([[0.28,0.52,0.34],[0.35,0.74,0.44],[0.32,0.96,0.48],[0.16,1.08,0.39],[-0.06,1.05,0.25],[-0.24,0.93,0.12]],0.082,10),[]);
  const svc=useMemo(()=>mkTube([[0.56,1.28,-0.05],[0.56,1.02,0.02],[0.54,0.76,0.04]],0.07,10),[]);
  const ivc=useMemo(()=>mkTube([[0.56,-0.42,-0.1],[0.53,-0.08,-0.02],[0.5,0.2,0.03]],0.064,10),[]);
  const lpv=useMemo(()=>mkTube([[-0.74,0.86,-0.26],[-0.44,0.78,-0.16],[-0.26,0.74,-0.1]],0.052,8),[]);
  const rpv=useMemo(()=>mkTube([[0.68,0.84,-0.22],[0.42,0.78,-0.14],[0.26,0.74,-0.1]],0.052,8),[]);
  const rca=useMemo(()=>mkTube([[0.42,0.46,0.23],[0.58,0.34,0.16],[0.66,0.16,0.02],[0.62,-0.1,-0.12],[0.4,-0.36,-0.16]],0.02,7),[]);
  const lad=useMemo(()=>mkTube([[0.02,0.58,0.46],[0.02,0.3,0.54],[-0.02,0.04,0.52],[-0.08,-0.22,0.44],[-0.16,-0.56,0.2]],0.019,7),[]);
  const lcx=useMemo(()=>mkTube([[0.0,0.62,0.44],[-0.24,0.58,0.26],[-0.46,0.46,0.08],[-0.58,0.28,-0.1],[-0.58,0.08,-0.24]],0.018,7),[]);
  const diag=useMemo(()=>mkTube([[0.0,0.36,0.52],[0.18,0.18,0.48],[0.26,-0.08,0.34],[0.22,-0.34,0.12]],0.013,6),[]);
  const marg=useMemo(()=>mkTube([[0.52,0.28,0.2],[0.66,0.12,0.06],[0.66,-0.08,-0.08],[0.58,-0.3,-0.18]],0.013,6),[]);
  const post=useMemo(()=>mkTube([[-0.38,0.18,-0.08],[-0.54,0.0,-0.2],[-0.54,-0.22,-0.26],[-0.46,-0.44,-0.18]],0.012,6),[]);
  const bct=useMemo(()=>mkTube([[0.26,1.16,0.2],[0.34,1.36,0.14],[0.42,1.52,0.1]],0.045,8),[]);
  const lcc=useMemo(()=>mkTube([[0.14,1.16,0.24],[0.16,1.38,0.24],[0.18,1.56,0.26]],0.037,8),[]);
  const lsa=useMemo(()=>mkTube([[0.04,1.14,0.24],[-0.02,1.32,0.3],[-0.14,1.46,0.34],[-0.28,1.56,0.34]],0.041,8),[]);
  const cSinus=useMemo(()=>mkTube([[-0.46,0.26,-0.2],[-0.3,0.18,-0.28],[-0.12,0.1,-0.32],[0.08,0.02,-0.3]],0.018,7),[]);
  return <group scale={isRat ? 0.66 : 0.92} position={isRat ? [0.02,0.06,0.12] : [0,0.03,0.16]}>
    <HoloMesh geo={aGeo} color={color} frag={VESSEL_FRAG} opacity={0.95}/>
    <HoloMesh geo={pGeo} color={color} frag={VESSEL_FRAG} opacity={0.88}/>
    <HoloMesh geo={svc} color={shiftBlue(color)} frag={VESSEL_FRAG} opacity={0.82}/>
    <HoloMesh geo={ivc} color={shiftBlue(color)} frag={VESSEL_FRAG} opacity={0.82}/>
    <HoloMesh geo={lpv} color={shiftBlue(color)} frag={VESSEL_FRAG} opacity={0.75}/>
    <HoloMesh geo={rpv} color={shiftBlue(color)} frag={VESSEL_FRAG} opacity={0.75}/>
    <HoloMesh geo={rca} color={brightenC(color,1.3)} frag={VESSEL_FRAG} opacity={0.92}/>
    <HoloMesh geo={lad} color={brightenC(color,1.3)} frag={VESSEL_FRAG} opacity={0.92}/>
    <HoloMesh geo={lcx} color={brightenC(color,1.3)} frag={VESSEL_FRAG} opacity={0.88}/>
    <HoloMesh geo={diag} color={brightenC(color,1.28)} frag={VESSEL_FRAG} opacity={0.86}/>
    <HoloMesh geo={marg} color={brightenC(color,1.25)} frag={VESSEL_FRAG} opacity={0.84}/>
    <HoloMesh geo={post} color={brightenC(color,1.22)} frag={VESSEL_FRAG} opacity={0.82}/>
    <HoloMesh geo={bct} color={color} frag={VESSEL_FRAG} opacity={0.86}/>
    <HoloMesh geo={lcc} color={color} frag={VESSEL_FRAG} opacity={0.84}/>
    <HoloMesh geo={lsa} color={color} frag={VESSEL_FRAG} opacity={0.84}/>
    <HoloMesh geo={cSinus} color={shiftBlue(color)} frag={VESSEL_FRAG} opacity={0.72}/>
  </group>;
}

function HeartValves({ color, isRat }) {
  const ringColor = brightenC(color, 1.22);
  const valves = isRat
    ? [
        { id:"aortic", p:[0.04,0.56,0.21], r:[1.4,0.2,0], t:0.13 },
        { id:"pulmonary", p:[0.2,0.55,0.28], r:[1.3,-0.3,0], t:0.12 },
        { id:"tricuspid", p:[0.18,0.32,0.04], r:[1.55,0.1,0], t:0.14 },
        { id:"mitral", p:[-0.1,0.34,-0.03], r:[1.52,-0.08,0], t:0.12 },
      ]
    : [
        { id:"aortic", p:[0.06,0.62,0.24], r:[1.35,0.2,0], t:0.16 },
        { id:"pulmonary", p:[0.26,0.6,0.34], r:[1.25,-0.32,0], t:0.15 },
        { id:"tricuspid", p:[0.24,0.36,0.07], r:[1.55,0.12,0], t:0.18 },
        { id:"mitral", p:[-0.13,0.38,-0.02], r:[1.52,-0.08,0], t:0.16 },
      ];
  return <group scale={isRat ? 0.76 : 1}>
    {valves.map((v)=><group key={v.id} position={v.p} rotation={v.r}>
      <Torus args={[v.t, v.t * 0.12, 8, 36]}>
        <meshStandardMaterial color={ringColor} emissive={ringColor} emissiveIntensity={0.7} transparent opacity={0.8} />
      </Torus>
      {[0,1,2].map((i)=>{
        const a = (i * Math.PI * 2) / 3;
        return <mesh key={`${v.id}-${i}`} position={[Math.cos(a) * v.t * 0.34, 0, Math.sin(a) * v.t * 0.34]} rotation={[0, a, 0]}>
          <coneGeometry args={[v.t * 0.09, 0.06, 6]} />
          <meshStandardMaterial color={ringColor} emissive={ringColor} emissiveIntensity={0.52} transparent opacity={0.7} />
        </mesh>;
      })}
      {[0,1,2].map((i)=>{
        const a = (i * Math.PI * 2) / 3;
        const tip=[Math.cos(a) * v.t * 0.34,-0.03,Math.sin(a) * v.t * 0.34];
        const end=[Math.cos(a) * v.t * 0.14,-0.2,Math.sin(a) * v.t * 0.14];
        const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(...tip),new THREE.Vector3(...end)]);
        return <line key={`${v.id}-c-${i}`} geometry={g}><lineBasicMaterial color={ringColor} transparent opacity={0.45} /></line>;
      })}
    </group>)}
  </group>;
}

function HeartMuscleLayer({ color, geo, isRat }) {
  const outer = isRat ? "#74101c" : "#7e1020";
  const inner = isRat ? "#a51f31" : "#b21f37";
  return <group scale={isRat ? 0.78 : 1}>
    <mesh geometry={geo}>
      <meshPhysicalMaterial color={outer} emissive={outer} emissiveIntensity={0.18} roughness={0.38} metalness={0.04} transmission={0} clearcoat={0.3} clearcoatRoughness={0.5} />
    </mesh>
    <mesh geometry={geo} scale={[0.985,0.985,0.985]}>
      <meshStandardMaterial color={inner} emissive={color} emissiveIntensity={0.08} transparent opacity={0.34} roughness={0.42} metalness={0.08} />
    </mesh>
  </group>;
}

function HeartMeshOverlay({ color, geo, isRat }) {
  return <group scale={isRat ? 0.78 : 1}>
    <mesh geometry={geo} scale={[1.01,1.01,1.01]}>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.2} transparent opacity={0.12} wireframe depthWrite={false} />
    </mesh>
    <mesh geometry={geo} scale={[0.994,0.994,0.994]}>
      <meshStandardMaterial color={brightenC(color,1.2)} emissive={color} emissiveIntensity={0.1} transparent opacity={0.08} wireframe depthWrite={false} />
    </mesh>
  </group>;
}

function HeartBeatGroup({ color, isRat }) {
  const ref=useRef(); const geo=useMemo(()=>buildHeart(isRat),[isRat]);
  useFrame(({clock})=>{
    if(!ref.current)return;
    const freq=isRat?2.4:1.1;
    const ph=(clock.elapsedTime*freq)%(Math.PI*2);
    const comp=isRat?0.05:0.055;
    let s;
    if(ph<0.4)s=1-Math.sin(ph/0.4*Math.PI)*comp;
    else if(ph<1.2)s=(1-comp)+((ph-0.4)/0.8)*comp;
    else s=1;
    ref.current.scale.setScalar(s*(isRat?0.78:1));
    ref.current.rotation.y=Math.sin(ph)*(isRat?0.014:0.018);
  });
  return <group ref={ref}>
    <HeartMuscleLayer color={color} geo={geo} isRat={isRat}/>
    <HeartMeshOverlay color={color} geo={geo} isRat={isRat}/>
    <HoloMesh geo={geo} color={color} frag={BEAT_FRAG} opacity={0.72}/>
    <HeartValves color={color} isRat={isRat}/>
    <HeartVessels color={color} isRat={isRat}/>
  </group>;
}

// 
// HUMAN BODY SCENE (HOMEPAGE)
// 
const LEGACY_ORGAN_HOTSPOTS = {
  brain: { color: "#00E5FF", pos: [0, 2.08, 0.02], label: "BRAIN" },
  heart: { color: "#FF1A3C", pos: [-0.14, 1.08, 0.22], label: "HEART" },
  liver: { color: "#FF6B35", pos: [0.22, 0.62, 0.18], label: "LIVER" },
  lungs: { color: "#88CCFF", pos: [0, 0.98, 0.18], label: "LUNGS" },
  spine: { color: "#FFD700", pos: [0.0, 0.5, -0.08], label: "SPINE" },
  dna: { color: "#FF3EAA", pos: [0.38, -0.1, 0.12], label: "GENOME" },
};
const ATLAS_ORGAN_HOTSPOTS = {
  brain: { color: "#AA44FF", pos: [0.0, 1.45, 0.05], label: "BRAIN" },
  heart: { color: "#FF2244", pos: [-0.18, 0.55, 0.12], label: "HEART" },
  liver: { color: "#FF6600", pos: [0.22, 0.0, 0.1], label: "LIVER" },
};

function OrganBeacon({ color, active=false }) {
  const coreGeo = useMemo(() => new THREE.SphereGeometry(0.028, 10, 10), []);
  const ringGeo = useMemo(() => new THREE.TorusGeometry(0.07, 0.006, 6, 20), []);
  return (
    <group scale={active ? 1.35 : 1}>
      <mesh geometry={coreGeo}>
        <meshBasicMaterial color={color} transparent opacity={active ? 1 : 0.96} />
      </mesh>
      <mesh geometry={ringGeo}>
        <meshBasicMaterial color={color} transparent opacity={active ? 0.92 : 0.66} />
      </mesh>
      <mesh geometry={ringGeo} rotation={[Math.PI / 2, 0, 0]}>
        <meshBasicMaterial color={color} transparent opacity={active ? 0.92 : 0.66} />
      </mesh>
    </group>
  );
}

function BodyScene({ highlight, onCursorOrganChange }) {
  const gRef = useRef();
  const C = "#00E5FF";
  const [atlas, setAtlas] = useState(null);
  const [atlasError, setAtlasError] = useState(false);
  const [cursorOrgan, setCursorOrgan] = useState(null);
  useEffect(() => {
    let cancelled = false;
    loadHumanAtlasModel()
      .then((model) => {
        if (cancelled) return;
        setAtlas(model);
        setAtlasError(false);
      })
      .catch((err) => {
        console.error("Human atlas model load failed:", err);
        if (cancelled) return;
        setAtlas(null);
        setAtlasError(true);
      });
    return () => { cancelled = true; };
  }, []);
  const atlasWireGeo = useMemo(() => (atlas?.geometry ? atlas.geometry.clone() : null), [atlas]);
  const atlasShellGeo = useMemo(() => (atlas?.geometry ? atlas.geometry.clone() : null), [atlas]);
  const atlasPointsGeo = useMemo(() => {
    const src = atlas?.geometry?.attributes?.position;
    if (!src) return null;
    const step = 18;
    const count = Math.floor(src.count / step);
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const si = i * step;
      arr[i * 3] = src.getX(si);
      arr[i * 3 + 1] = src.getY(si);
      arr[i * 3 + 2] = src.getZ(si);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    return g;
  }, [atlas]);
  const atlasReady = Boolean(atlas && atlasWireGeo && atlasShellGeo);
  const hotspots = ATLAS_ORGAN_HOTSPOTS;
  const hoveredRef = useRef(null);
  const lastMoveTsRef = useRef(0);

  useFrame(({ clock }) => { if (gRef.current) gRef.current.rotation.y = Math.sin(clock.elapsedTime * 0.07) * 0.1; });

  const activeOrgan = cursorOrgan || highlight;
  const oc = (id) => (activeOrgan === id ? hotspots[id]?.color || C : C);
  const setHoveredOrgan = (id) => {
    if (hoveredRef.current === id) return;
    hoveredRef.current = id;
    setCursorOrgan(id);
    if (onCursorOrganChange) onCursorOrganChange(id);
    document.body.style.cursor = id ? "pointer" : "crosshair";
  };
  const detectCursorOrgan = (worldPoint) => {
    if (!gRef.current) return null;
    const p = gRef.current.worldToLocal(worldPoint.clone());
    let nearest = null;
    let bestD2 = Infinity;
    const radii = { brain: 0.62, heart: 0.42, liver: 0.46 };
    for (const [id, { pos }] of Object.entries(hotspots)) {
      const dx = p.x - pos[0];
      const dy = p.y - pos[1];
      const dz = p.z - pos[2];
      const d2 = dx * dx + dy * dy + dz * dz;
      const r = radii[id] || 0.45;
      if (d2 <= r * r && d2 < bestD2) {
        bestD2 = d2;
        nearest = id;
      }
    }
    return nearest;
  };
  const onModelPointerMove = (e) => {
    e.stopPropagation();
    if (e.timeStamp - lastMoveTsRef.current < 32) return;
    lastMoveTsRef.current = e.timeStamp;
    const id = detectCursorOrgan(e.point);
    setHoveredOrgan(id);
  };
  const clearCursorHighlight = () => {
    setHoveredOrgan(null);
  };
  useEffect(() => () => {
    document.body.style.cursor = "crosshair";
    if (onCursorOrganChange) onCursorOrganChange(null);
  }, [onCursorOrganChange]);

  return (
    <group ref={gRef} position={[0, 0, 0]} scale={1}>
      <ambientLight intensity={0.1} />
      <pointLight position={[3, 6, 4]} intensity={3} color={C} />
      <pointLight position={[-3, 2, -4]} intensity={1.5} color={C} />
      <pointLight position={[0, -2, 6]} intensity={1} color="#ffffff" />

      {atlasReady ? (
        <>
          <group rotation={[0, Math.PI, 0]}>
            <mesh geometry={atlasWireGeo} scale={[atlas.scale, atlas.scale, atlas.scale]} onPointerMove={onModelPointerMove} onPointerOut={clearCursorHighlight}>
              <meshBasicMaterial color={C} wireframe transparent opacity={1} />
            </mesh>
            <mesh geometry={atlasShellGeo} scale={[atlas.scale, atlas.scale, atlas.scale]} onPointerMove={onModelPointerMove} onPointerOut={clearCursorHighlight}>
              <meshBasicMaterial color={C} transparent opacity={0.22} side={THREE.DoubleSide} depthWrite={false} />
            </mesh>
            <points geometry={atlasPointsGeo} scale={[atlas.scale, atlas.scale, atlas.scale]}>
              <pointsMaterial color={C} size={0.008} transparent opacity={0.95} sizeAttenuation />
            </points>
          </group>
          {Object.keys(hotspots).map((id) => (
            <group key={id} position={hotspots[id].pos}>
              <mesh
                onPointerOver={(e)=>{e.stopPropagation();setHoveredOrgan(id);}}
                onPointerOut={(e)=>{e.stopPropagation();setHoveredOrgan(null);}}
              >
                <sphereGeometry args={[id==="brain" ? 0.34 : 0.24, 12, 12]} />
                <meshBasicMaterial transparent opacity={0} depthWrite={false} />
              </mesh>
              <OrganBeacon color={oc(id)} active={activeOrgan === id} />
            </group>
          ))}
        </>
      ) : (
        <group>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <torusGeometry args={[0.9, 0.012, 10, 84]} />
            <meshBasicMaterial color={atlasError ? "#FF3344" : C} transparent opacity={0.85} />
          </mesh>
          <mesh rotation={[Math.PI / 2, 0.8, 0]}>
            <torusGeometry args={[0.55, 0.01, 10, 84]} />
            <meshBasicMaterial color={C} transparent opacity={0.7} />
          </mesh>
        </group>
      )}
      {atlasReady && Object.entries(hotspots).map(([id,{color,pos,label}])=>(
        <group key={id} position={pos}>
          <mesh><sphereGeometry args={[0.018,8,8]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={1.5}/></mesh>
          <mesh position={[0.2,0,0]}><boxGeometry args={[0.25,0.01,0.01]}/><meshBasicMaterial color={color} transparent opacity={0.7}/></mesh>
        </group>
      ))}
    </group>
  );
}

// 
// HUD + PANELS (shared)
// 
const cornerBracket = (color, pos) => {
  const top=pos.includes("top"),btm=pos.includes("bottom"),lft=pos.includes("left"),rgt=pos.includes("right");
  return <div style={{position:"absolute",top:top?12:"auto",bottom:btm?12:"auto",left:lft?12:"auto",right:rgt?12:"auto",width:28,height:28,borderTop:top?`2px solid ${color}55`:"none",borderBottom:btm?`2px solid ${color}55`:"none",borderLeft:lft?`2px solid ${color}55`:"none",borderRight:rgt?`2px solid ${color}55`:"none",transition:"border-color 0.8s ease"}}/>;
};

// 
// PAGE COMPONENTS
// 

const ADMET_API_BASE = (import.meta.env.VITE_ADMET_API_URL || "http://127.0.0.1:5051").replace(/\/+$/, "");
const ENDPOINT_ORDER = ["BBB", "DILI", "Ames", "hERG"];
const ENDPOINT_TO_ORGAN = { BBB: "brain", DILI: "liver", Ames: "genetic", hERG: "heart" };
const ENDPOINT_COLORS = { BBB: "#AA44FF", DILI: "#FF6600", Ames: "#FF3EAA", hERG: "#FF2244" };
const RISK_CATEGORY_COLORS = { Low: "#00ff88", Moderate: "#ffcc00", High: "#ff4d6d" };
const ATLAS_THEME = { none: "#39FF88", normal: "#00A8FF", high: "#FF365E" };

const fmtPercent = (p) => `${(Math.max(0, Math.min(1, Number(p) || 0)) * 100).toFixed(1)}%`;
const fmtScore = (p) => (Number.isFinite(Number(p)) ? Number(p).toFixed(3) : "0.000");
const riskFromProb = (p) => {
  const v = Number(p) || 0;
  if (v >= 0.7) return "HIGH";
  if (v >= 0.3) return "MODERATE";
  return "LOW";
};

function getAtlasTheme(summary) {
  if (!summary) return { state: "NONE", color: ATLAS_THEME.none };
  if (summary.risk_category === "High") return { state: "HIGH", color: ATLAS_THEME.high };
  return { state: "NORMAL", color: ATLAS_THEME.normal };
}

function buildHomeOrganInfo(payload) {
  if (!payload?.organs) return {};
  const map = {
    brain: { title: "BRAIN", endpoint: "BBB" },
    heart: { title: "HEART", endpoint: "hERG" },
    liver: { title: "LIVER", endpoint: "DILI" },
  };
  const out = {};
  Object.entries(map).forEach(([key, info]) => {
    const d = payload.organs[key];
    if (!d) return;
    const pct = Number(d.percent || 0);
    out[key] = {
      title: info.title,
      color: d.risk_color || ENDPOINT_COLORS[info.endpoint] || "#00E5FF",
      status: `${info.endpoint} RISK: ${d.risk_label || riskFromProb(d.probability)}`,
      metricLabel: `${info.endpoint} PROBABILITY`,
      metricValue: Math.max(0, Math.min(100, Math.round(pct))),
      detail: d.message || "No message",
    };
  });
  return out;
}

function DNASidePanel({ color, datasetReference, lastUpdated, predictionCount }) {
  const rungs = Array.from({ length: 12 }, (_, i) => {
    const y = 8 + i * 8;
    const phase = i * 0.55;
    const x1 = 26 + Math.sin(phase) * 9;
    const x2 = 74 - Math.sin(phase) * 9;
    return { i, y, x1, x2 };
  });
  return (
    <div style={{padding:"10px 12px",border:`1px solid ${color}18`,borderRadius:4,background:`${color}03`}}>
      <div style={{color:`${color}66`,fontSize:8,letterSpacing:2,marginBottom:8}}>DNA STRUCTURE</div>
      <svg viewBox="0 0 100 108" style={{width:"100%",height:104,display:"block"}}>
        <path d="M28,6 C12,22 12,38 28,54 C44,70 44,86 28,102" fill="none" stroke="#55B8FF" strokeWidth="2.2" strokeOpacity="0.9"/>
        <path d="M72,6 C88,22 88,38 72,54 C56,70 56,86 72,102" fill="none" stroke="#FF76A4" strokeWidth="2.2" strokeOpacity="0.9"/>
        {rungs.map(({i,y,x1,x2})=>(
          <g key={i}>
            <line x1={x1} y1={y} x2={x2} y2={y} stroke="#c4f6ff" strokeOpacity="0.6" strokeWidth="1.4"/>
            <circle cx={x1} cy={y} r="1.5" fill="#55B8FF"/>
            <circle cx={x2} cy={y} r="1.5" fill="#FF76A4"/>
          </g>
        ))}
      </svg>
      <div style={{display:"flex",justifyContent:"space-between",marginTop:6,fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>
        <span style={{color:"#55B8FF"}}>A/T</span>
        <span style={{color:`${color}66`}}>DOUBLE HELIX</span>
        <span style={{color:"#FF76A4"}}>G/C</span>
      </div>
      <div style={{marginTop:8,paddingTop:6,borderTop:`1px solid ${color}18`,fontFamily:"'Share Tech Mono',monospace",fontSize:8,lineHeight:1.55}}>
        <div style={{display:"flex",justifyContent:"space-between"}}>
          <span style={{color:`${color}66`}}>TRAIN FP</span>
          <span style={{color:"#d7fbff"}}>{datasetReference?.training_fingerprints ?? "N/A"}</span>
        </div>
        <div style={{display:"flex",justifyContent:"space-between"}}>
          <span style={{color:`${color}66`}}>RUNS</span>
          <span style={{color:"#d7fbff"}}>{predictionCount}</span>
        </div>
        <div style={{display:"flex",justifyContent:"space-between"}}>
          <span style={{color:`${color}66`}}>UPDATED</span>
          <span style={{color:"#d7fbff"}}>{lastUpdated ? lastUpdated.toLocaleTimeString() : "N/A"}</span>
        </div>
      </div>
    </div>
  );
}

function useLiveAdmet(initialSmiles = "CCO") {
  const [smiles, setSmiles] = useState(initialSmiles);
  const [autoLive, setAutoLive] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [errorText, setErrorText] = useState("");
  const [apiHealth, setApiHealth] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [runCount, setRunCount] = useState(0);

  const checkHealth = useCallback(async () => {
    try {
      const r = await fetch(`${ADMET_API_BASE}/health`);
      const d = await r.json();
      setApiHealth(d);
    } catch {
      setApiHealth({ ok: false, load_error: "ADMET API unreachable" });
    }
  }, []);

  const runPrediction = useCallback(async (nextSmiles) => {
    const cleaned = String(nextSmiles || "").trim();
    if (!cleaned) {
      setErrorText("Enter a SMILES string");
      return;
    }
    setPredicting(true);
    setErrorText("");
    try {
      const r = await fetch(`${ADMET_API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: cleaned, include_shap: false }),
      });
      const d = await r.json().catch(() => ({ ok: false, error: "Invalid API response" }));
      if (!r.ok || !d.ok) throw new Error(d?.error || "Prediction failed");
      setPrediction(d);
      setLastUpdated(new Date());
      setRunCount((v) => v + 1);
    } catch (err) {
      setPrediction(null);
      setErrorText(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setPredicting(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const id = setInterval(checkHealth, 15000);
    return () => clearInterval(id);
  }, [checkHealth]);

  useEffect(() => {
    if (!autoLive) return;
    const trimmed = smiles.trim();
    if (trimmed.length < 2) return;
    const id = setTimeout(() => { runPrediction(trimmed); }, 550);
    return () => clearTimeout(id);
  }, [smiles, autoLive, runPrediction]);

  const activePrediction = prediction?.ok ? prediction : null;
  const summary = activePrediction?.summary || null;
  const endpointInfo = activePrediction?.endpoint_info || {};
  const datasetReference = activePrediction?.dataset_reference || {};
  const riskCategory = summary?.risk_category || "N/A";
  const riskColor = RISK_CATEGORY_COLORS[riskCategory] || "#00E5FF";
  const adStatus = activePrediction?.raw?.ad_status || "Unknown";
  const adSimilarity = activePrediction?.raw?.ad_similarity;
  const adText = Number.isFinite(adSimilarity) ? `${adStatus} (${Number(adSimilarity).toFixed(3)})` : adStatus;
  const endpointRows = ENDPOINT_ORDER.map((endpoint) => {
    const prob = Number(activePrediction?.predictions?.[endpoint] || 0);
    const organKey = ENDPOINT_TO_ORGAN[endpoint];
    const organData = activePrediction?.organs?.[organKey];
    const color = organData?.risk_color || ENDPOINT_COLORS[endpoint];
    const label = endpointInfo?.[endpoint]?.full_name || endpoint;
    return {
      endpoint,
      label,
      modelType: endpointInfo?.[endpoint]?.type || "Model",
      score: fmtScore(prob),
      pct: fmtPercent(prob),
      bar: `${Math.round(prob * 100)}`,
      color,
      risk: organData?.risk_label || riskFromProb(prob),
      message: organData?.message || "No signal message",
    };
  });
  const riskFlags = activePrediction?.risk_flags?.length
    ? activePrediction.risk_flags
    : [errorText ? `Prediction error: ${errorText}` : "Run prediction to generate organ risk flags."];
  const recommendation = summary?.recommendation || "Awaiting prediction output.";
  const systemOnline = Boolean(apiHealth?.ok);

  return {
    smiles,
    setSmiles,
    autoLive,
    setAutoLive,
    predicting,
    runPrediction,
    errorText,
    apiHealth,
    lastUpdated,
    runCount,
    activePrediction,
    summary,
    datasetReference,
    riskCategory,
    riskColor,
    adStatus,
    adText,
    endpointRows,
    riskFlags,
    recommendation,
    systemOnline,
  };
}

function ModelRiskPanel({ color, live, focus }) {
  const focusEndpoint = { brain: "BBB", heart: "hERG", liver: "DILI", dna: "Ames" }[focus] || "BBB";
  const primary = live.endpointRows.find((r) => r.endpoint === focusEndpoint);
  return (
    <div style={{position:"absolute",right:18,top:126,zIndex:28,width:255,padding:"10px 12px",fontFamily:"'Orbitron',monospace",background:"rgba(2,10,20,0.9)",border:`1px solid ${color}88`,boxShadow:`0 0 26px ${color}55, inset 0 0 18px ${color}22`,backdropFilter:"blur(6px)"}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:7}}>
        <span style={{color,fontSize:9,letterSpacing:2.4}}>LIVE PREDICTION PANEL</span>
        <span style={{color:live.systemOnline ? "#39FF88" : "#ff6f8f",fontSize:8,letterSpacing:1.2}}>{live.systemOnline ? "API ONLINE" : "API OFFLINE"}</span>
      </div>
      <div style={{display:"flex",gap:6,marginBottom:8}}>
        <input
          value={live.smiles}
          onChange={(e)=>live.setSmiles(e.target.value)}
          spellCheck={false}
          placeholder="SMILES"
          style={{flex:1,height:28,background:"#061120",border:`1px solid ${color}44`,color:"#d7fbff",padding:"0 8px",fontSize:9,fontFamily:"'Share Tech Mono',monospace",outline:"none"}}
        />
        <button
          onClick={()=>live.runPrediction(live.smiles)}
          disabled={live.predicting}
          style={{width:72,height:28,border:`1px solid ${color}99`,background:live.predicting?`${color}22`:`${color}14`,color,fontSize:8,letterSpacing:1.3,fontFamily:"'Orbitron',monospace",cursor:"crosshair"}}
        >
          {live.predicting ? "WAIT" : "RUN"}
        </button>
      </div>
      <div style={{marginBottom:8,padding:"6px 7px",border:`1px solid ${primary?.color || color}77`,background:`${primary?.color || color}1a`,boxShadow:`0 0 18px ${(primary?.color || color)}55`}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"baseline"}}>
          <span style={{color:primary?.color || color,fontSize:10,letterSpacing:1.6,fontWeight:700}}>{focusEndpoint} FOCUS</span>
          <span style={{color:"#ffffff",fontSize:16,letterSpacing:1.2,fontWeight:900,textShadow:`0 0 18px ${primary?.color || color}`}}>{primary?.pct || "N/A"}</span>
        </div>
        <div style={{marginTop:5,height:5,background:"#09111d",overflow:"hidden"}}>
          <div style={{height:"100%",width:`${primary?.bar || 0}%`,background:`linear-gradient(to right,${color}88,${primary?.color || color})`,boxShadow:`0 0 14px ${primary?.color || color}`}}/>
        </div>
      </div>
      {live.endpointRows.map((row)=>(
        <div key={`model-risk-${row.endpoint}`} style={{padding:"5px 6px",marginBottom:4,border:`1px solid ${row.endpoint === focusEndpoint ? row.color + "99" : color + "24"}`,background:row.endpoint === focusEndpoint ? `${row.color}22` : `${color}08`,boxShadow:row.endpoint === focusEndpoint ? `0 0 14px ${row.color}44` : "none"}}>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
            <span style={{color:row.color,fontSize:8.5,letterSpacing:1.2}}>{row.endpoint} . {row.risk}</span>
            <span style={{color:"#d7fbff",fontSize:9,fontFamily:"'Share Tech Mono',monospace"}}>{row.score}</span>
          </div>
          <div style={{height:3,marginTop:3,background:"#0a1018"}}>
            <div style={{height:"100%",width:`${row.bar}%`,background:`linear-gradient(to right,${row.color}66,${row.color})`}}/>
          </div>
        </div>
      ))}
      <div style={{marginTop:5,color:live.errorText ? "#ff6f8f" : `${color}99`,fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>
        {live.errorText || `AD ${live.adText} | ${live.summary ? `COMPOSITE ${Number(live.summary.composite_score).toFixed(3)}` : "Awaiting run"}`}
      </div>
    </div>
  );
}

//  HOME 
function HomePage({ onNav, live }) {
  const [hov, setHov] = useState(null);
  const [cursorOrgan, setCursorOrgan] = useState(null);
  const [tm, setTm] = useState(new Date());
  useEffect(()=>{const t=setInterval(()=>setTm(new Date()),1000);return()=>clearInterval(t);},[]);
  const C="#00E5FF";
  const {
    smiles, setSmiles, autoLive, setAutoLive, predicting, runPrediction, errorText,
    lastUpdated, runCount, activePrediction, summary, datasetReference, riskCategory,
    riskColor, adStatus, adText, endpointRows, riskFlags, recommendation, systemOnline,
  } = live;
  const atlasTheme = getAtlasTheme(summary);
  const atlasThemeColor = atlasTheme.color;
  const organPopupData = buildHomeOrganInfo(activePrediction);
  const organPopup = organPopupData[cursorOrgan] || null;

  const bbb = endpointRows.find((e) => e.endpoint === "BBB");
  const dili = endpointRows.find((e) => e.endpoint === "DILI");
  const ames = endpointRows.find((e) => e.endpoint === "Ames");
  const herg = endpointRows.find((e) => e.endpoint === "hERG");
  const cards = [
    { id:"brain", label:"NEURAL ATLAS", sub:"Brain Anatomy", color:"#00E5FF", icon:"NE", stat:bbb ? `${bbb.pct} BBB` : "No data" },
    { id:"heart", label:"CARDIAC ATLAS", sub:"Heart Anatomy", color:"#FF1A3C", icon:"CA", stat:herg ? `${herg.pct} hERG` : "No data" },
    { id:"liver", label:"HEPATIC ATLAS", sub:"Liver Anatomy", color:"#FF6B35", icon:"HE", stat:dili ? `${dili.pct} DILI` : "No data" },
    { id:"dna", label:"GENOMIC ATLAS", sub:"DNA Structure", color:"#FF3EAA", icon:"GN", stat:ames ? `${ames.pct} AMES` : "No data" },
  ];

  return (
    <div style={{width:"100%",height:"100%",position:"relative",overflow:"hidden"}}>
      {/* Scan line */}
      <div style={{position:"absolute",left:0,right:0,height:1,background:`linear-gradient(to right,transparent,${C}44 30%,${C}44 70%,transparent)`,animation:"scanLine 7s linear infinite",zIndex:2,pointerEvents:"none"}}/>

      {/* CENTER 3D BODY */}
      <div style={{position:"absolute",left:"50%",top:0,bottom:32,width:640,transform:"translateX(-50%)",zIndex:5}}>
        <Canvas camera={{position:[0,0.5,6.5],fov:44}} dpr={[1,1]} gl={{antialias:false,alpha:true,powerPreference:"high-performance"}} style={{background:"transparent",width:"100%",height:"100%"}} onPointerMissed={()=>setCursorOrgan(null)}>
          <BodyScene highlight={hov} onCursorOrganChange={setCursorOrgan}/>
          <points>
            <bufferGeometry><bufferAttribute attach="attributes-position" count={260} array={useMemo(()=>{const a=new Float32Array(780);for(let i=0;i<260;i++){a[i*3]=(Math.random()-0.5)*10;a[i*3+1]=(Math.random()-0.5)*12;a[i*3+2]=(Math.random()-0.5)*8;}return a;},[])} itemSize={3}/></bufferGeometry>
            <pointsMaterial size={0.015} color={C} transparent opacity={0.3} sizeAttenuation/>
          </points>
          <OrbitControls enablePan={false} minDistance={3.5} maxDistance={11}/>
        </Canvas>
      </div>
      {organPopup && (
        <div style={{position:"absolute",left:"50%",top:132,transform:"translateX(-50%)",zIndex:30,pointerEvents:"none",fontFamily:"'Orbitron',monospace",minWidth:270,padding:"12px 14px",background:"rgba(3,15,32,0.92)",border:`1px solid ${organPopup.color}88`,boxShadow:`0 0 24px ${organPopup.color}33`,backdropFilter:"blur(6px)"}}>
          <div style={{color:`${C}88`,fontSize:8,letterSpacing:2,marginBottom:6}}>ORGAN FOCUS</div>
          <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:8}}>
            <span style={{color:organPopup.color,fontSize:14,letterSpacing:2,fontWeight:700}}>{organPopup.title}</span>
            <span style={{color:"#39FF88",fontSize:9,letterSpacing:1}}>LIVE</span>
          </div>
          <div style={{color:`${C}88`,fontSize:9,letterSpacing:1.2,lineHeight:1.6}}>{organPopup.metricLabel}: {organPopup.metricValue}%</div>
          <div style={{color:`${C}88`,fontSize:9,letterSpacing:1.2,lineHeight:1.6}}>{organPopup.status}</div>
          <div style={{color:`${C}88`,fontSize:9,letterSpacing:1.2,lineHeight:1.6}}>{organPopup.detail}</div>
          <div style={{color:`${C}88`,fontSize:9,letterSpacing:1.2,lineHeight:1.6}}>SPECIMEN: HOMO SAPIENS</div>
          <div style={{marginTop:7,height:2,background:"#0b1a2a"}}>
            <div style={{height:"100%",width:`${organPopup.metricValue}%`,background:`linear-gradient(to right,${organPopup.color}66,${organPopup.color})`}}/>
          </div>
        </div>
      )}

      {/* TITLE */}
      <div style={{position:"absolute",top:74,left:"50%",transform:"translateX(-50%)",zIndex:20,textAlign:"center",fontFamily:"'Orbitron',monospace",pointerEvents:"none"}}>
        <div style={{color:C,fontSize:9,letterSpacing:5,opacity:0.5,marginBottom:3}}>HOLOGRAPHIC ANATOMY SYSTEM</div>
        <div style={{color:C,fontSize:22,letterSpacing:7,fontWeight:900,textShadow:`0 0 35px ${C},0 0 70px ${C}44`}}>HUMAN ATLAS</div>
        <div style={{color:`${C}44`,fontSize:8,letterSpacing:3,marginTop:3}}>FULL-BODY ANATOMICAL PROJECTION - DRAG TO ROTATE</div>
      </div>

      {/* LEFT PANEL */}
      <div style={{position:"absolute",left:14,top:70,bottom:38,width:258,zIndex:10,display:"flex",flexDirection:"column",gap:8,fontFamily:"'Orbitron',monospace",border:`1px solid ${atlasThemeColor}26`,padding:"10px 10px 8px",background:"rgba(2,10,20,0.35)",boxShadow:`0 0 24px ${atlasThemeColor}22`}}>
        <div style={{color:atlasThemeColor,fontSize:8,letterSpacing:3,opacity:0.9}}>MODEL STACK</div>
        {endpointRows.map((row)=>(
          <div key={row.endpoint} style={{display:"flex",justifyContent:"space-between",alignItems:"center",padding:"6px 8px",border:`1px solid ${atlasThemeColor}33`,borderRadius:3,background:`${atlasThemeColor}0d`}}>
            <span style={{color:`${C}66`,fontSize:7.5,letterSpacing:1.2}}>{row.endpoint} / {row.modelType}</span>
            <span style={{color:row.color,fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>{row.risk}</span>
          </div>
        ))}
        <div style={{marginTop:6,color:atlasThemeColor,fontSize:8,letterSpacing:3,opacity:0.92}}>LIVE PREDICTION %</div>
        {endpointRows.map((row)=>{
          const rowTone = !activePrediction ? ATLAS_THEME.none : (Number(row.bar) >= 70 ? ATLAS_THEME.high : ATLAS_THEME.normal);
          const rowLevel = !activePrediction ? "NONE" : (Number(row.bar) >= 70 ? "HIGH" : "LOW");
          return (
          <div key={`${row.endpoint}-bar`} style={{marginBottom:5,padding:"5px 6px",border:`1px solid ${rowTone}55`,background:`${rowTone}14`,boxShadow:`0 0 14px ${rowTone}33`}}>
            <div style={{display:"flex",justifyContent:"space-between",marginBottom:2}}>
              <span style={{color:`${C}55`,fontSize:7.5,letterSpacing:1}}>{row.endpoint}</span>
              <span style={{color:rowTone,fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>{row.pct} . {rowLevel}</span>
            </div>
            <div style={{height:3,background:"#0a1015",borderRadius:2,overflow:"hidden"}}>
              <div style={{height:"100%",width:`${row.bar}%`,background:`linear-gradient(to right,${rowTone}66,${rowTone})`,boxShadow:`0 0 14px ${rowTone}`}}/>
            </div>
          </div>
        )})}
        <div style={{marginTop:8,padding:"8px 10px",border:`1px solid ${atlasThemeColor}45`,borderRadius:3,background:`${atlasThemeColor}10`,boxShadow:`0 0 16px ${atlasThemeColor}33`}}>
          <div style={{color:C,fontSize:11,letterSpacing:2,fontWeight:700}}>INPUT SUMMARY</div>
          <div style={{color:`${C}55`,fontSize:7.5,letterSpacing:1.5,marginTop:3}}>SMILES: <span style={{color:"#d7fbff"}}>{smiles || "N/A"}</span></div>
          <div style={{color:`${C}55`,fontSize:7.5,letterSpacing:1.5,marginTop:2}}>RISK: <span style={{color:riskColor}}>{riskCategory}</span></div>
          <div style={{color:`${C}55`,fontSize:7.5,letterSpacing:1.5,marginTop:2}}>AD STATUS: <span style={{color:"#d7fbff"}}>{adText}</span></div>
          <div style={{color:`${C}44`,fontSize:7.5,marginTop:2}}>COMPOSITE: <span style={{color:"#d7fbff"}}>{summary ? Number(summary.composite_score).toFixed(3) : "N/A"}</span></div>
        </div>
        <div style={{color:`${C}33`,fontSize:7.5,fontFamily:"'Share Tech Mono',monospace"}}>{tm.toLocaleString()} | API {systemOnline ? "ONLINE" : "OFFLINE"}</div>
      </div>

      {/* RIGHT PANEL */}
      <div style={{position:"absolute",right:14,top:70,bottom:38,width:280,zIndex:10,display:"flex",flexDirection:"column",gap:10,fontFamily:"'Orbitron',monospace",border:`1px solid ${atlasThemeColor}26`,padding:"10px 10px 8px",background:"rgba(2,10,20,0.35)",boxShadow:`0 0 24px ${atlasThemeColor}22`}}>
        <div style={{padding:"8px 10px",border:`1px solid ${atlasThemeColor}55`,borderRadius:4,background:`${atlasThemeColor}10`,boxShadow:`0 0 18px ${atlasThemeColor}33`}}>
          <div style={{color:C,fontSize:8,letterSpacing:3,opacity:0.7,marginBottom:7}}>SMILES INPUT</div>
          <input
            value={smiles}
            onChange={(e)=>setSmiles(e.target.value)}
            onKeyDown={(e)=>{ if (e.key === "Enter") runPrediction(smiles); }}
            spellCheck={false}
            placeholder="Enter SMILES (e.g., CCO)"
            style={{width:"100%",height:30,background:"rgba(1,8,16,0.8)",border:`1px solid ${C}33`,color:"#d7fbff",padding:"0 8px",fontSize:10,fontFamily:"'Share Tech Mono',monospace",outline:"none"}}
          />
          <div style={{display:"flex",gap:6,marginTop:7}}>
            <button onClick={()=>runPrediction(smiles)} disabled={predicting}
              style={{flex:1,height:26,border:`1px solid ${atlasThemeColor}99`,background:predicting?`${atlasThemeColor}24`:`${atlasThemeColor}12`,color:atlasThemeColor,fontSize:9,letterSpacing:1.5,fontFamily:"'Orbitron',monospace",cursor:"crosshair"}}>
              {predicting ? "RUNNING..." : "RUN PREDICT"}
            </button>
            <button onClick={()=>setAutoLive((v)=>!v)}
              style={{width:76,height:26,border:`1px solid ${autoLive ? "#39FF88" : C+"55"}`,background:autoLive?"rgba(57,255,136,0.15)":`${C}08`,color:autoLive?"#39FF88":C,fontSize:8,letterSpacing:1,fontFamily:"'Orbitron',monospace",cursor:"crosshair"}}>
              {autoLive ? "AUTO ON" : "AUTO OFF"}
            </button>
          </div>
          <div style={{marginTop:6,fontSize:8,fontFamily:"'Share Tech Mono',monospace",color:errorText ? "#ff6f8f" : `${C}66`}}>
            {errorText || `ADMET API: ${systemOnline ? "ONLINE" : "OFFLINE"}${lastUpdated ? ` | Updated ${lastUpdated.toLocaleTimeString()}` : ""}`}
          </div>
        </div>

        <div style={{padding:"8px 10px",border:`1px solid ${atlasThemeColor}44`,borderRadius:4,background:`${atlasThemeColor}10`,boxShadow:`0 0 16px ${atlasThemeColor}22`}}>
          <div style={{color:`${C}55`,fontSize:7.5,letterSpacing:2,marginBottom:6}}>ORGAN RISK FLAGS</div>
          {riskFlags.map((flag, i)=>(
            <div key={`${flag}-${i}`} style={{fontSize:8.5,lineHeight:1.5,color:`${C}88`,marginBottom:5,border:`1px solid ${C}14`,padding:"5px 6px",background:`${C}03`}}>{flag}</div>
          ))}
        </div>

        <div style={{padding:"8px 10px",border:`1px solid ${atlasThemeColor}44`,borderRadius:4,background:`${atlasThemeColor}10`,boxShadow:`0 0 16px ${atlasThemeColor}22`}}>
          <div style={{color:`${C}55`,fontSize:7.5,letterSpacing:2,marginBottom:5}}>RECOMMENDATION</div>
          <div style={{fontSize:8.5,lineHeight:1.6,color:riskColor}}>{recommendation}</div>
        </div>

        <div style={{color:atlasThemeColor,fontSize:8,letterSpacing:3,opacity:0.95}}>ANATOMY MODULES</div>
        {cards.map(({id,label,sub,color,icon,stat})=>(
          <button key={id} onClick={()=>onNav(id)} onMouseEnter={()=>setHov(id)} onMouseLeave={()=>setHov(null)}
            style={{padding:"12px 13px",background:hov===id?`${atlasThemeColor}22`:`${atlasThemeColor}0d`,border:`1px solid ${hov===id?atlasThemeColor:atlasThemeColor+"77"}`,borderRadius:4,cursor:"crosshair",textAlign:"left",transition:"all 0.25s ease",boxShadow:hov===id?`0 0 26px ${atlasThemeColor}66`:`0 0 16px ${atlasThemeColor}26`,transform:hov===id?"translateX(-4px)":"none",fontFamily:"'Orbitron',monospace"}}>
            <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:5}}>
              <span style={{color:atlasThemeColor,fontSize:13,textShadow:`0 0 12px ${atlasThemeColor}`}}>{icon}</span>
              <span style={{color:atlasThemeColor,fontSize:9,letterSpacing:2,fontWeight:700}}>{label}</span>
            </div>
            <div style={{color:`${atlasThemeColor}cc`,fontSize:8,letterSpacing:1.5,marginBottom:3}}>{sub}</div>
            <div style={{height:1,background:`linear-gradient(to right,${atlasThemeColor}66,transparent)`,marginBottom:5}}/>
            <div style={{color:`${atlasThemeColor}dd`,fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>{stat}</div>
          </button>
        ))}
        <DNASidePanel color={C} datasetReference={datasetReference} lastUpdated={lastUpdated} predictionCount={runCount}/>
      </div>

      {/* BOTTOM BAR */}
      <div style={{position:"absolute",bottom:0,left:0,right:0,height:32,zIndex:20,background:"rgba(2,8,16,0.9)",borderTop:`1px solid ${C}10`,display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 24px",fontFamily:"'Share Tech Mono',monospace"}}>
        <span style={{color:`${C}44`,fontSize:8,letterSpacing:2}}>
          COMPOSITE {summary ? Number(summary.composite_score).toFixed(3) : "N/A"} | BBB {bbb?.pct || "N/A"} | DILI {dili?.pct || "N/A"} | AMES {ames?.pct || "N/A"} | hERG {herg?.pct || "N/A"}
        </span>
        <span style={{color:`${C}33`,fontSize:8,letterSpacing:1}}>DRAG TO ROTATE | HOVER ORGANS FOR LIVE POPUP</span>
        <span style={{color:atlasThemeColor,fontSize:8,letterSpacing:2}}>LEVEL {atlasTheme.state === "NONE" ? "NONE" : atlasTheme.state === "HIGH" ? "HIGH" : "LOW"} | RISK {riskCategory} | AD {adStatus}</span>
      </div>
    </div>
  );
}
function BrainPage({ live }) {
  const [mode,setMode]=useState("human");const [tr,setTr]=useState(false);
  const isH=mode==="human";const color=isH?"#00E5FF":"#39FF88";
  const sw=m=>{if(m===mode)return;setTr(true);setTimeout(()=>{setMode(m);setTr(false);},400);};
  const geo=useMemo(()=>{
    const g = buildBrain(!isH);
    if(!g || !g.attributes?.position) return new THREE.SphereGeometry(isH?1.05:0.85,72,72);
    const arr = g.attributes.position.array;
    for(let i=0;i<arr.length;i++){ if(!Number.isFinite(arr[i])) return new THREE.SphereGeometry(isH?1.05:0.85,72,72); }
    return g;
  },[isH]);
  const gRef=useRef();
  useEffect(()=>{setTr(true);setTimeout(()=>setTr(false),400);},[isH]);
  return (
    <div style={{width:"100%",height:"100%",background:"#020810",backgroundImage:`radial-gradient(ellipse at 50% 50%,${isH?"rgba(0,229,255,0.04)":"rgba(57,255,136,0.04)"} 0%,transparent 70%)`,overflow:"hidden",position:"relative",transition:"background 0.8s"}}>
      <div style={{position:"absolute",inset:0,opacity:0.065,backgroundImage:`linear-gradient(${color}44 1px,transparent 1px),linear-gradient(90deg,${color}44 1px,transparent 1px)`,backgroundSize:"40px 40px",transition:"all 0.8s"}}/>
      <Toggle color={color} isH={isH} sw={sw} labelA="HUMAN BRAIN" labelB="RAT BRAIN"/>
      <div style={{width:"100%",height:"100%",opacity:tr?0:1,transition:"opacity 0.4s"}}>
        <Canvas camera={{position:[0,0,4],fov:50}} gl={{antialias:true,alpha:true}} style={{background:"transparent"}}>
          <ambientLight intensity={0.15}/>
          <pointLight position={[4,4,4]} intensity={1.5} color={color}/>
          <pointLight position={[-4,-2,-4]} intensity={0.8} color={color}/>
          <group ref={gRef}>
            <BrainRotate geo={geo} color={color} isH={isH}/>
          </group>
          <Particles color={color} count={1200} spread={1.8}/>
          <ScanRings color={color}/>
          <MRIPlane color={color}/>
          <OrbitControls enablePan={false} minDistance={2.5} maxDistance={6}/>
        </Canvas>
      </div>
      <OrganHUD color={color} title="NEURAL ATLAS v4.2" scan="HOLOGRAPHIC PROJECTION"
        species={isH?"HOMO SAPIENS":"RATTUS NORVEGICUS"} organ="BRAIN"
        specs={isH?{mass:"1,300g",neurons:"86B",synapses:"150T",vol:"1,260 cm3"}:{mass:"2.0g",neurons:"71M",synapses:"500B",vol:"2.2 cm3"}}
        structs={isH?["Cerebrum","Cerebellum","Brainstem","Corpus Callosum"]:["Olfactory Bulb","Cortex","Cerebellum","Brainstem"]}
        specimen={isH?"HUMAN":"RODENT"}
      />
      <ModelRiskPanel color={color} live={live} focus="brain" />
    </div>
  );
}
function BrainRotate({geo,color,isH}){const r=useRef();useFrame(({clock})=>{if(r.current){r.current.rotation.y=clock.elapsedTime*0.11;r.current.position.y=Math.sin(clock.elapsedTime*0.7)*0.03;}});return <group ref={r} scale={isH?1:0.92}><mesh geometry={geo}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.55} transparent opacity={0.24} metalness={0.18} roughness={0.35} depthWrite={false}/></mesh><HoloMesh geo={geo} color={color} opacity={0.92}/><mesh geometry={geo}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.22} transparent opacity={0.14} wireframe depthWrite={false}/></mesh><BrainLobes color={color} isRat={!isH}/></group>;}

//  HEART PAGE 
function HeartPage({ live }) {
  const [mode,setMode]=useState("human");const [tr,setTr]=useState(false);
  const isH=mode==="human";const color=isH?"#FF1A3C":"#FF6B1A";
  const sw=m=>{if(m===mode)return;setTr(true);setTimeout(()=>{setMode(m);setTr(false);},450);};
  const gRef=useRef();
  return (
    <div style={{width:"100%",height:"100%",background:"#03050a",backgroundImage:`radial-gradient(ellipse at 50% 45%,${isH?"rgba(255,26,60,0.06)":"rgba(255,107,26,0.05)"} 0%,transparent 65%)`,overflow:"hidden",position:"relative",transition:"background 0.9s"}}>
      <div style={{position:"absolute",inset:0,opacity:0.05,backgroundImage:`linear-gradient(${color}33 1px,transparent 1px),linear-gradient(90deg,${color}33 1px,transparent 1px)`,backgroundSize:"38px 38px"}}/>
      <Toggle color={color} isH={isH} sw={sw} labelA="HUMAN HEART" labelB="RAT HEART"/>
      <div style={{width:"100%",height:"100%",opacity:tr?0:1,transition:"opacity 0.45s"}}>
        <Canvas camera={{position:[0,0.18,4.7],fov:46}} gl={{antialias:true,alpha:true}} style={{background:"transparent"}}>
          <ambientLight intensity={0.06}/>
          <pointLight position={[3.6,4.8,3.8]} intensity={2.1} color={color}/>
          <pointLight position={[-3.8,-1.8,-3.6]} intensity={0.7} color={color}/>
          <pointLight position={[0,0,6]} intensity={0.95} color="#ffffff"/>
          <directionalLight position={[1.5,2.4,2]} intensity={0.7} color="#ffd0d0" />
          <group ref={gRef}>
            <HeartRotate color={color} isH={isH}/>
          </group>
          <Particles color={color} count={960} spread={1.45} beat/>
          <ScanRings color={color} radii={[1.9,2.14,2.38]} speed={[0.22,0.16,0.1]}/>
          <OrbitControls enablePan={false} minDistance={2.8} maxDistance={8} enableDamping dampingFactor={0.06}/>
        </Canvas>
      </div>
      <OrganHUD color={color} title="CARDIAC ATLAS v5.0" scan="REAL-TIME SCAN"
        species={isH?"HOMO SAPIENS":"RATTUS NORVEGICUS"} organ="HEART"
        specs={isH?{weight:"250-350g",output:"5 L/min",rate:"60-100 bpm",pressure:"120/80"}:{weight:"0.9-1.4g",output:"22 mL/min",rate:"250-450 bpm",pressure:"116/90"}}
        structs={["Aortic Valve","Mitral Valve","Tricuspid Valve","Pulmonary Valve","Coronary Arteries","Interventricular Septum"]}
        specimen={isH?"HUMAN":"RODENT"} ecg
      />
      <ModelRiskPanel color={color} live={live} focus="heart" />
    </div>
  );
}
function HeartRotate({color,isH}){const r=useRef();useFrame(({clock})=>{if(r.current)r.current.rotation.y=clock.elapsedTime*(isH?0.08:0.1);});return <group ref={r}><HeartBeatGroup color={color} isRat={!isH}/></group>;}

//  LIVER PAGE 
function LiverPage({ live }) {
  const [mode,setMode]=useState("human");const [tr,setTr]=useState(false);
  const isH=mode==="human";const color=isH?"#FF6B35":"#A8FF3E";
  const sw=m=>{if(m===mode)return;setTr(true);setTimeout(()=>{setMode(m);setTr(false);},400);};
  const geo=useMemo(()=>buildLiver(!isH),[isH]);
  const gRef=useRef();
  return (
    <div style={{width:"100%",height:"100%",background:"#06080a",backgroundImage:`radial-gradient(ellipse at 50% 50%,${isH?"rgba(255,107,53,0.05)":"rgba(168,255,62,0.04)"} 0%,transparent 70%)`,overflow:"hidden",position:"relative"}}>
      <div style={{position:"absolute",inset:0,opacity:0.06,backgroundImage:`linear-gradient(${color}55 1px,transparent 1px),linear-gradient(90deg,${color}55 1px,transparent 1px)`,backgroundSize:"40px 40px"}}/>
      <Toggle color={color} isH={isH} sw={sw} labelA="HUMAN LIVER" labelB="RAT LIVER"/>
      <div style={{width:"100%",height:"100%",opacity:tr?0:1,transition:"opacity 0.4s"}}>
        <Canvas camera={{position:[0,0,4],fov:50}} gl={{antialias:true,alpha:true}} style={{background:"transparent"}}>
          <ambientLight intensity={0.12}/>
          <pointLight position={[4,4,4]} intensity={1.8} color={color}/>
          <pointLight position={[-4,-2,-4]} intensity={0.9} color={color}/>
          <group ref={gRef}>
            <LiverRotate geo={geo} color={color}/>
          </group>
          <Particles color={color} count={900} spread={1.5}/>
          <ScanRings color={color}/>
          <MRIPlane color={color} range={0.85}/>
          <OrbitControls enablePan={false} minDistance={2.5} maxDistance={7}/>
        </Canvas>
      </div>
      <OrganHUD color={color} title="HEPATIC ATLAS v2.7" scan="METABOLIC SCAN"
        species={isH?"HOMO SAPIENS":"RATTUS NORVEGICUS"} organ="LIVER"
        specs={isH?{weight:"1,500g",lobes:"4",vol:"1,470 cm3",func:"500+"}:{weight:"10g",lobes:"6",vol:"9.8 cm3",func:"500+"}}
        structs={isH?["Right Lobe","Left Lobe","Gallbladder","Portal Vein","Hepatic Vein"]:["R.Lateral Lobe","L.Lateral Lobe","Median Lobe","Caudate Lobe","L.Medial Lobe"]}
        enzymes={isH?[{n:"ALT",v:32,m:56},{n:"AST",v:28,m:40},{n:"ALP",v:74,m:120},{n:"GGT",v:22,m:60}]:[{n:"ALT",v:45,m:80},{n:"AST",v:90,m:150},{n:"ALP",v:90,m:180},{n:"GGT",v:12,m:40}]}
        specimen={isH?"HUMAN":"RODENT"}
      />
      <ModelRiskPanel color={color} live={live} focus="liver" />
    </div>
  );
}
function LiverRotate({geo,color}){const r=useRef();useFrame(({clock})=>{if(r.current)r.current.rotation.y=clock.elapsedTime*0.1;});return <group ref={r}><HoloMesh geo={geo} color={color} opacity={1}/></group>;}

function DNAHelix({ color, dna, stream }) {
  const gRef = useRef();
  const pRef = useRef();
  const pairGeo = useMemo(() => new THREE.BoxGeometry(1, 0.03, 0.064), []);
  const sugarGeo = useMemo(() => new THREE.CylinderGeometry(0.024, 0.024, 0.075, 10), []);
  const phosphateGeo = useMemo(() => new THREE.OctahedronGeometry(0.03, 0), []);

  useFrame(({ clock }) => {
    if (gRef.current) gRef.current.rotation.y = clock.elapsedTime * 0.16;
    if (!pRef.current) return;
    const arr = pRef.current.geometry.attributes.position.array;
    for (let i = 0; i < stream.count; i++) {
      arr[i * 3 + 1] += stream.sp[i] * 0.0042;
      if (arr[i * 3 + 1] > dna.height / 2) arr[i * 3 + 1] = -dna.height / 2;
    }
    pRef.current.geometry.attributes.position.needsUpdate = true;
  });

  return <>
    <group ref={gRef}>
      <mesh geometry={dna.b1}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} transparent opacity={0.86} /></mesh>
      <mesh geometry={dna.b2}><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.9} transparent opacity={0.86} /></mesh>
      {dna.pairs.map((p) => <group key={p.i} position={p.mid} quaternion={p.quat}>
        <mesh geometry={pairGeo} scale={[p.len, 1, 1]}>
          <meshStandardMaterial color={p.baseColor} emissive={p.baseColor} emissiveIntensity={0.74} transparent opacity={0.9} />
        </mesh>
        <mesh geometry={sugarGeo} position={[-p.len * 0.54, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <meshStandardMaterial color={p.sugarColor} emissive={p.sugarColor} emissiveIntensity={0.5} transparent opacity={0.85} />
        </mesh>
        <mesh geometry={sugarGeo} position={[p.len * 0.54, 0, 0]} rotation={[0, 0, Math.PI / 2]}>
          <meshStandardMaterial color={p.sugarColor} emissive={p.sugarColor} emissiveIntensity={0.5} transparent opacity={0.85} />
        </mesh>
        <mesh geometry={phosphateGeo} position={[-p.len * 0.72, 0, 0]}>
          <meshStandardMaterial color={p.phosphateColor} emissive={p.phosphateColor} emissiveIntensity={0.62} transparent opacity={0.88} />
        </mesh>
        <mesh geometry={phosphateGeo} position={[p.len * 0.72, 0, 0]}>
          <meshStandardMaterial color={p.phosphateColor} emissive={p.phosphateColor} emissiveIntensity={0.62} transparent opacity={0.88} />
        </mesh>
      </group>)}
    </group>
    <points ref={pRef}>
      <bufferGeometry><bufferAttribute attach="attributes-position" count={stream.count} array={stream.arr} itemSize={3} /></bufferGeometry>
      <pointsMaterial size={0.013} color={color} transparent opacity={0.56} sizeAttenuation />
    </points>
  </>;
}

//  DNA PAGE 
function DNAPage({ live }) {
  const [mode,setMode]=useState("human");const [tr,setTr]=useState(false);
  const isH=mode==="human";const color=isH?"#00CFFF":"#FF3EAA";
  const sw=m=>{if(m===mode)return;setTr(true);setTimeout(()=>{setMode(m);setTr(false);},400);};
  const BPC=useMemo(()=>isH?["#F46A6A","#55B8FF","#F8D552","#68E989"]:["#FF76A4","#8CE7B0","#FFB15A","#B5A0FF"],[isH]);
  const dna = useMemo(() => {
    const turns = isH ? 10.6 : 8.1;
    const pitchBoost = 3.8;
    const height = (isH ? 6.4 : 5.0) * pitchBoost;
    const baseRadius = isH ? 0.94 : 0.8;
    const pairCount = isH ? 118 : 88;
    const steps = pairCount * 5;
    const a = [];
    const b = [];
    const axis = new THREE.Vector3(1, 0, 0);
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const ang = t * Math.PI * 2 * turns;
      const y = t * height - height / 2;
      const groove = Math.sin(ang * 2.0) * 0.09;
      const ra = baseRadius * (1 + groove);
      const rb = baseRadius * (1 - groove * 0.82);
      a.push(new THREE.Vector3(Math.cos(ang) * ra, y, Math.sin(ang) * ra));
      b.push(new THREE.Vector3(Math.cos(ang + Math.PI) * rb, y, Math.sin(ang + Math.PI) * rb));
    }
    const b1 = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(a), steps, isH ? 0.033 : 0.03, 12, false);
    const b2 = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(b), steps, isH ? 0.033 : 0.03, 12, false);
    const pairs = [];
    for (let i = 0; i < pairCount; i++) {
      const t = i / (pairCount - 1);
      const ang = t * Math.PI * 2 * turns;
      const y = t * height - height / 2;
      const groove = Math.sin(ang * 2.0) * 0.09;
      const p1 = new THREE.Vector3(Math.cos(ang) * (baseRadius * (1 + groove)), y, Math.sin(ang) * (baseRadius * (1 + groove)));
      const p2 = new THREE.Vector3(Math.cos(ang + Math.PI) * (baseRadius * (1 - groove * 0.82)), y, Math.sin(ang + Math.PI) * (baseRadius * (1 - groove * 0.82)));
      const mid = p1.clone().add(p2).multiplyScalar(0.5);
      const dir = p2.clone().sub(p1);
      const len = dir.length() * 0.84;
      dir.normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(axis, dir);
      const twist = new THREE.Quaternion().setFromAxisAngle(dir, Math.sin(ang * 1.45) * 0.22);
      q.multiply(twist);
      pairs.push({
        i,
        mid: [mid.x, mid.y, mid.z],
        quat: [q.x, q.y, q.z, q.w],
        len,
        baseColor: BPC[i % 4],
        sugarColor: i % 2 ? "#83C8FF" : "#6BB6FF",
        phosphateColor: i % 2 ? "#FF7BC8" : "#EC64C0",
      });
    }
    return { height, b1, b2, pairs };
  }, [isH, BPC]);
  const stream = useMemo(() => {
    const count = 1500;
    const arr = new Float32Array(count * 3);
    const sp = new Float32Array(count);
    const spread = isH ? 1.85 : 1.55;
    for (let i = 0; i < count; i++) {
      const t = Math.random();
      const ang = t * Math.PI * 20;
      const rr = 0.75 + Math.random() * spread;
      arr[i * 3] = Math.cos(ang) * rr;
      arr[i * 3 + 1] = t * dna.height - dna.height / 2;
      arr[i * 3 + 2] = Math.sin(ang) * rr;
      sp[i] = 0.18 + Math.random() * 0.55;
    }
    return { count, arr, sp };
  }, [dna.height, isH]);
  return (
    <div style={{width:"100%",height:"100%",background:"#04060a",backgroundImage:`radial-gradient(ellipse at 50% 50%,${isH?"rgba(0,207,255,0.05)":"rgba(255,62,170,0.05)"} 0%,transparent 70%)`,overflow:"hidden",position:"relative"}}>
      <div style={{position:"absolute",inset:0,opacity:0.055,backgroundImage:`linear-gradient(${color}44 1px,transparent 1px),linear-gradient(90deg,${color}44 1px,transparent 1px)`,backgroundSize:"36px 36px"}}/>
      <Toggle color={color} isH={isH} sw={sw} labelA="HUMAN DNA" labelB="RAT DNA"/>
      <div style={{width:"100%",height:"100%",opacity:tr?0:1,transition:"opacity 0.4s"}}>
        <Canvas camera={{position:[0,0,8.6],fov:52}} gl={{antialias:true,alpha:true}} style={{background:"transparent"}}>
          <ambientLight intensity={0.11} />
          <pointLight position={[5,5,5]} intensity={2} color={color} />
          <pointLight position={[-5,-3,-5]} intensity={1} color={color} />
          <DNAHelix color={color} dna={dna} stream={stream} />
          <ScanRings color={color} radii={[isH?2.2:1.8,isH?2.45:2.05,isH?2.7:2.3]} />
          <OrbitControls enablePan={false} minDistance={3.4} maxDistance={12} />
        </Canvas>
      </div>
      <OrganHUD color={color} title="GENOMIC ATLAS v3.1" scan="SEQUENCE SCAN ACTIVE"
        species={isH?"HOMO SAPIENS":"RATTUS NORVEGICUS"} organ="DNA HELIX"
        specs={isH?{chromosomes:"46 (23 pairs)",basePairs:"3.2 Billion",genes:"~20,000",genome:"3.2 GB"}:{chromosomes:"42 (21 pairs)",basePairs:"2.75 Billion",genes:"~22,000",genome:"2.75 GB"}}
        structs={["Sugar-Phosphate Backbone","Hydrogen Bond Planes","Major Groove","Minor Groove","B-Form Helix"]}
        specimen={isH?"HUMAN":"RODENT"}
        bases={isH?[{n:"A",p:30.9,c:"#F46A6A"},{n:"T",p:29.4,c:"#F8D552"},{n:"G",p:19.9,c:"#55B8FF"},{n:"C",p:19.8,c:"#68E989"}]:[{n:"A",p:29.4,c:"#FF76A4"},{n:"T",p:28.8,c:"#FFB15A"},{n:"G",p:21.2,c:"#8CE7B0"},{n:"C",p:20.6,c:"#B5A0FF"}]}
      />
      <ModelRiskPanel color={color} live={live} focus="dna" />
    </div>
  );
}

//  SHARED TOGGLE 
function Toggle({ color, isH, sw, labelA, labelB }) {
  return (
    <div style={{position:"absolute",top:68,left:"50%",transform:"translateX(-50%)",zIndex:10,display:"flex",alignItems:"center",fontFamily:"'Orbitron',monospace"}}>
      <button onClick={()=>sw("human")} style={{padding:"10px 28px",background:isH?color:"transparent",border:`1px solid ${isH?color:color+"44"}`,color:isH?"#020810":`${color}88`,fontSize:11,letterSpacing:3,fontFamily:"inherit",cursor:"crosshair",fontWeight:isH?700:400,boxShadow:isH?`0 0 30px ${color}66`:"none",transition:"all 0.4s ease",borderRight:"none",borderRadius:"4px 0 0 4px"}}>{labelA}</button>
      <div style={{width:40,height:38,background:"#0a1520",border:`1px solid ${color}22`,display:"flex",alignItems:"center",justifyContent:"center"}}><div style={{width:10,height:10,borderRadius:"50%",background:color,boxShadow:`0 0 16px ${color}`,animation:"pulse-dot 1.5s ease-in-out infinite"}}/></div>
      <button onClick={()=>sw("rat")} style={{padding:"10px 28px",background:!isH?color:"transparent",border:`1px solid ${!isH?color:color+"44"}`,color:!isH?"#020810":`${color}88`,fontSize:11,letterSpacing:3,fontFamily:"inherit",cursor:"crosshair",fontWeight:!isH?700:400,boxShadow:!isH?`0 0 30px ${color}66`:"none",transition:"all 0.4s ease",borderLeft:"none",borderRadius:"0 4px 4px 0"}}>{labelB}</button>
    </div>
  );
}

//  SHARED ORGAN HUD 
function OrganHUD({ color, title, scan, species, organ, specs, structs, specimen, ecg=false, enzymes, bases }) {
  return (
    <div style={{position:"absolute",inset:0,pointerEvents:"none",fontFamily:"'Orbitron',monospace"}}>
      {["topleft","topright","bottomleft","bottomright"].map(pos=>(
        <div key={pos} style={{position:"absolute",top:pos.includes("top")?12:"auto",bottom:pos.includes("bottom")?12:"auto",left:pos.includes("left")?12:"auto",right:pos.includes("right")?12:"auto",width:28,height:28,borderTop:pos.includes("top")?`2px solid ${color}55`:"none",borderBottom:pos.includes("bottom")?`2px solid ${color}55`:"none",borderLeft:pos.includes("left")?`2px solid ${color}55`:"none",borderRight:pos.includes("right")?`2px solid ${color}55`:"none",transition:"border-color 0.8s ease"}}/>
      ))}
      <div style={{position:"absolute",top:0,left:0,right:0,padding:"16px 24px",display:"flex",justifyContent:"space-between",alignItems:"center",borderBottom:`1px solid ${color}22`}}>
        <div style={{display:"flex",alignItems:"center",gap:12}}><div style={{width:8,height:8,borderRadius:"50%",background:color,boxShadow:`0 0 14px ${color}`,animation:`${ecg?"heart-beat":"pulse-dot"} ${ecg?"0.85":"1.5"}s ease-in-out infinite`}}/><span style={{color,fontSize:11,letterSpacing:3}}>{title}</span></div>
        <span style={{color,fontSize:10,letterSpacing:2,opacity:0.6}}>{scan}</span>
        <span style={{color,fontSize:10,letterSpacing:2,opacity:0.5}}>{new Date().toLocaleTimeString()}</span>
      </div>
      <div style={{position:"absolute",top:80,left:24,maxWidth:268}}>
        <div style={{color,fontSize:9,letterSpacing:3,opacity:0.5,marginBottom:2}}>{species}</div>
        <div style={{color,fontSize:20,letterSpacing:4,fontWeight:900,textShadow:`0 0 28px ${color}`,marginBottom:8}}>{organ}</div>
        {Object.entries(specs).map(([k,v])=><div key={k} style={{display:"flex",gap:10,alignItems:"center",marginBottom:4}}><span style={{color:`${color}66`,fontSize:8,letterSpacing:1.5,width:78,textTransform:"uppercase"}}>{k}</span><div style={{width:55,height:1,background:`linear-gradient(to right,${color}44,transparent)`}}/><span style={{color,fontSize:10}}>{v}</span></div>)}
        {enzymes&&<><div style={{marginTop:10,color,fontSize:8,letterSpacing:3,opacity:0.5,marginBottom:5}}>ENZYME LEVELS</div>{enzymes.map(({n,v,m})=><div key={n} style={{display:"flex",alignItems:"center",gap:8,marginBottom:4}}><span style={{color:`${color}99`,fontSize:8,width:28}}>{n}</span><div style={{width:96,height:4,background:"#0a1015",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:`${(v/m)*100}%`,background:`linear-gradient(to right,${color}88,${color})`,borderRadius:2}}/></div><span style={{color,fontSize:8}}>{v}</span></div>)}</>}
        {bases&&<><div style={{marginTop:10,color,fontSize:8,letterSpacing:3,opacity:0.5,marginBottom:5}}>BASE COMPOSITION</div>{bases.map(({n,p,c})=><div key={n} style={{display:"flex",alignItems:"center",gap:7,marginBottom:4}}><span style={{color:c,fontSize:9,width:14,fontWeight:700}}>{n}</span><div style={{width:80,height:4,background:"#0a1015",borderRadius:2,overflow:"hidden"}}><div style={{height:"100%",width:`${p*3}%`,background:`linear-gradient(to right,${c}88,${c})`,borderRadius:2}}/></div><span style={{color:c,fontSize:8}}>{p}%</span></div>)}</>}
        {ecg&&<><div style={{marginTop:10,color,fontSize:8,letterSpacing:3,opacity:0.5,marginBottom:3}}>ECG - NORMAL SINUS</div><div style={{border:`1px solid ${color}33`,borderRadius:4,padding:"3px 7px",background:`${color}06`}}><ECGLine color={color}/></div></>}
      </div>
      <div style={{position:"absolute",top:80,right:24,display:"flex",flexDirection:"column",alignItems:"flex-end",gap:7}}>
        <div style={{color,fontSize:8,letterSpacing:3,opacity:0.5}}>STRUCTURES</div>
        {structs.map((s,i)=><div key={s} style={{display:"flex",alignItems:"center",gap:10}}><span style={{color:`${color}99`,fontSize:9,letterSpacing:1}}>{s}</span><div style={{width:6,height:6,borderRadius:"50%",border:`1px solid ${color}`,background:color,boxShadow:`0 0 6px ${color}`,opacity:0.7+i*0.04}}/></div>)}
      </div>
      <div style={{position:"absolute",bottom:18,left:24,right:24,display:"flex",justifyContent:"space-between"}}>
        <span style={{color:`${color}55`,fontSize:8,letterSpacing:2}}>DRAG TO ROTATE  SCROLL TO ZOOM</span>
        <span style={{color:`${color}55`,fontSize:8,letterSpacing:2}}>{specimen} SPECIMEN SCAN</span>
      </div>
    </div>
  );
}

// 
// NAVBAR
// 
const NAV=[
  {id:"home",label:"ATLAS",icon:"AT",color:"#00E5FF"},
  {id:"brain",label:"NEURAL",icon:"NE",color:"#00E5FF"},
  {id:"heart",label:"CARDIAC",icon:"CA",color:"#FF1A3C"},
  {id:"liver",label:"HEPATIC",icon:"HE",color:"#FF6B35"},
  {id:"dna",label:"GENOMIC",icon:"GN",color:"#FF3EAA"},
];

function NavBar({ page, onNav }) {
  const [hov,setHov]=useState(null);
  const [tm,setTm]=useState(new Date());
  useEffect(()=>{const t=setInterval(()=>setTm(new Date()),1000);return()=>clearInterval(t);},[]);
  return (
    <div style={{position:"fixed",top:0,left:0,right:0,height:58,zIndex:1000,display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 22px",background:"rgba(2,8,16,0.96)",backdropFilter:"blur(14px)",borderBottom:"1px solid rgba(0,229,255,0.1)",fontFamily:"'Orbitron',monospace"}}>
      <div style={{display:"flex",alignItems:"center",gap:10}}>
        <div style={{width:32,height:32,border:"1px solid #00E5FF",borderRadius:"50%",display:"flex",alignItems:"center",justifyContent:"center",position:"relative"}}>
          <div style={{width:20,height:20,border:"1px solid rgba(0,229,255,0.3)",borderRadius:"50%",animation:"rotate-slow 4s linear infinite",position:"absolute"}}/>
          <span style={{color:"#00E5FF",fontSize:10,zIndex:1}}>BL</span>
        </div>
        <div>
          <div style={{color:"#00E5FF",fontSize:11,letterSpacing:4,fontWeight:700}}>BIOATLAS</div>
          <div style={{color:"rgba(0,229,255,0.4)",fontSize:7,letterSpacing:3}}>HOLOGRAPHIC LAB v2.0</div>
        </div>
      </div>
      <div style={{display:"flex",gap:5}}>
        {NAV.map(({id,label,icon,color})=>{
          const act=page===id,hv=hov===id;
          return <button key={id} onClick={()=>onNav(id)} onMouseEnter={()=>setHov(id)} onMouseLeave={()=>setHov(null)}
            style={{padding:"6px 17px",background:act?`${color}18`:"transparent",border:`1px solid ${act?color:color+"33"}`,color:act?color:`${color}77`,fontSize:9,letterSpacing:2.5,fontFamily:"inherit",cursor:"crosshair",borderRadius:3,transition:"all 0.2s ease",boxShadow:act?`0 0 18px ${color}33`:"none",transform:hv&&!act?"translateY(-1px)":"none",display:"flex",alignItems:"center",gap:5}}>
            <span style={{fontSize:10}}>{icon}</span>{label}
          </button>;
        })}
      </div>
      <div style={{display:"flex",alignItems:"center",gap:14}}>
        <div style={{display:"flex",alignItems:"center",gap:6}}><div style={{width:6,height:6,borderRadius:"50%",background:"#39FF88",boxShadow:"0 0 8px #39FF88",animation:"pulse-dot 2s ease-in-out infinite"}}/><span style={{color:"rgba(57,255,136,0.7)",fontSize:8,letterSpacing:2}}>ONLINE</span></div>
        <span style={{color:"rgba(0,229,255,0.3)",fontSize:8,fontFamily:"'Share Tech Mono',monospace"}}>{tm.toLocaleTimeString()}</span>
      </div>
    </div>
  );
}

// 
// ROOT APP
// 
export default function BioLabApp() {
  const [page,setPage]=useState("home");
  const live = useLiveAdmet("CCO");
  const pages={
    home:<HomePage onNav={setPage} live={live}/>,
    brain:<BrainPage live={live}/>,
    heart:<HeartPage live={live}/>,
    liver:<LiverPage live={live}/>,
    dna:<DNAPage live={live}/>,
  };
  return (
    <div style={{width:"100vw",height:"100vh",background:"#020810",backgroundImage:"radial-gradient(ellipse at 50% 50%,rgba(0,229,255,0.025) 0%,transparent 70%)",overflow:"hidden",position:"relative",cursor:"crosshair"}}>
      {/* Global grid */}
      <div style={{position:"absolute",inset:0,opacity:0.055,backgroundImage:"linear-gradient(rgba(0,229,255,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.06) 1px,transparent 1px)",backgroundSize:"42px 42px",pointerEvents:"none",zIndex:0}}/>
      <NavBar page={page} onNav={setPage}/>
      <div style={{position:"absolute",inset:0,top:58,zIndex:1}}>{pages[page]}</div>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@400;600&display=swap');
        *{box-sizing:border-box;margin:0;padding:0;}
        ::-webkit-scrollbar{width:3px;}::-webkit-scrollbar-thumb{background:#00E5FF;border-radius:2px;}
        @keyframes pulse-dot{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.3;transform:scale(0.7)}}
        @keyframes heart-beat{0%,100%{opacity:1;transform:scale(1)}40%{opacity:1;transform:scale(1.6)}50%{opacity:0.6;transform:scale(0.85)}}
        @keyframes rotate-slow{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
        @keyframes scanLine{from{transform:translateY(-100%)}to{transform:translateY(100vh)}}
      `}</style>
    </div>
  );
}




