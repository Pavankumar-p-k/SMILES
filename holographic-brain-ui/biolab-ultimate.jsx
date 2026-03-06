import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Torus } from "@react-three/drei";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

const mat = (c, o = 0.7, w = false) => <meshStandardMaterial color={c} emissive={c} emissiveIntensity={0.65} metalness={0.18} roughness={0.25} transparent opacity={o} wireframe={w} depthWrite={false} />;
const Holo = ({ g, c, o = 0.7, w = false }) => <mesh geometry={g}>{mat(c, o, w)}</mesh>;

function ellipsoid(rx, ry, rz, seg = 42) {
  const g = new THREE.SphereGeometry(1, seg, seg).toNonIndexed();
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) p.setXYZ(i, p.getX(i) * rx, p.getY(i) * ry, p.getZ(i) * rz);
  g.computeVertexNormals();
  return g;
}
function tube(pts, r = 0.05, radial = 8) {
  return new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map((p) => new THREE.Vector3(...p))), 36, r, radial, false);
}

function buildBrain(rat = false) {
  if (rat) {
    const c1 = new THREE.SphereGeometry(0.8, 84, 84).toNonIndexed();
    const c2 = new THREE.SphereGeometry(0.25, 40, 40).toNonIndexed();
    const stem = new THREE.CapsuleGeometry(0.07, 0.24, 6, 14).toNonIndexed();
    const p = c1.attributes.position;
    for (let i = 0; i < p.count; i++) {
      let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
      x *= 0.82; y *= 0.62; z *= 1.52;
      const a = Math.exp(-((z - 1.05) ** 2) / 0.09) * 0.2;
      const split = Math.exp(-((z - 1.02) ** 2) / 0.06) * 0.12;
      z += a; x += Math.sign(x || 1) * split * 0.45;
      const fiss = Math.exp(-(x * x) / 0.01) * Math.exp(-((y - 0.1) ** 2) / 0.24) * 0.08;
      x -= Math.sign(x || 1) * fiss;
      const gyri = Math.sin(z * 15 + y * 8) * 0.012 + Math.cos(x * 14 - z * 7) * 0.01;
      const s = 1 + gyri;
      p.setXYZ(i, x * s, y * s, z * s);
    }
    c2.translate(0, -0.22, -0.98); stem.rotateX(0.36); stem.translate(0, -0.46, -0.76);
    const m = mergeGeometries([c1, c2, stem], false); m.computeVertexNormals(); return m;
  }
  const c1 = new THREE.SphereGeometry(1.02, 96, 96).toNonIndexed();
  const c2 = new THREE.SphereGeometry(0.42, 56, 56).toNonIndexed();
  const stem = new THREE.CapsuleGeometry(0.11, 0.36, 8, 20).toNonIndexed();
  const p = c1.attributes.position;
  for (let i = 0; i < p.count; i++) {
    let x = p.getX(i), y = p.getY(i), z = p.getZ(i);
    x *= 1.18; y *= 0.92; z *= 1.06;
    const fiss = Math.exp(-(x * x) / 0.012) * Math.exp(-((y - 0.25) ** 2) / 0.5) * 0.16;
    x -= Math.sign(x || 1) * fiss;
    const gyri = Math.sin(z * 19 + y * 10) * 0.018 + Math.sin(x * 23 - z * 8) * 0.016 + Math.cos(y * 17 + x * 11) * 0.014;
    const s = 1 + gyri;
    p.setXYZ(i, x * s, y * s, z * s);
  }
  c2.translate(0, -0.5, -0.76); stem.rotateX(0.32); stem.translate(0, -0.83, -0.42);
  const m = mergeGeometries([c1, c2, stem], false); m.computeVertexNormals(); return m;
}

const H_NODES = { skull:[0,2,0], c7:[0,1.55,0], sternum:[0,1.05,0.18], t12:[0,0.5,-0.07], pelvis:[0,-0.52,0], shL:[0.72,1.2,0.05], shR:[-0.72,1.2,0.05], eL:[0.74,0.5,0.08], eR:[-0.74,0.5,0.08], wL:[0.72,-0.08,0.08], wR:[-0.72,-0.08,0.08], hL:[0.21,-0.52,0.03], hR:[-0.21,-0.52,0.03], kL:[0.2,-1.32,0.05], kR:[-0.2,-1.32,0.05], aL:[0.2,-2.05,0.04], aR:[-0.2,-2.05,0.04] };
const H_LINKS = [["skull","c7"],["c7","sternum"],["sternum","t12"],["t12","pelvis"],["c7","shL"],["shL","eL"],["eL","wL"],["c7","shR"],["shR","eR"],["eR","wR"],["pelvis","hL"],["hL","kL"],["kL","aL"],["pelvis","hR"],["hR","kR"],["kR","aR"]];
const R_NODES = { skull:[1.05,0.42,0], c1:[0.9,0.38,0], t1:[0.6,0.33,0], t6:[0.12,0.3,0], sac:[-0.74,0.22,0], t1a:[-0.96,0.2,0], t2a:[-1.2,0.18,0], t3a:[-1.45,0.15,0], scL:[0.45,0.32,0.16], scR:[0.45,0.32,-0.16], efL:[0.35,0.12,0.2], efR:[0.35,0.12,-0.2], wfL:[0.28,-0.12,0.22], wfR:[0.28,-0.12,-0.22], hhL:[-0.58,0.2,0.16], hhR:[-0.58,0.2,-0.16], khL:[-0.7,0.04,0.2], khR:[-0.7,0.04,-0.2], ahL:[-0.78,-0.12,0.22], ahR:[-0.78,-0.12,-0.22] };
const R_LINKS = [["skull","c1"],["c1","t1"],["t1","t6"],["t6","sac"],["sac","t1a"],["t1a","t2a"],["t2a","t3a"],["t1","scL"],["scL","efL"],["efL","wfL"],["t1","scR"],["scR","efR"],["efR","wfR"],["sac","hhL"],["hhL","khL"],["khL","ahL"],["sac","hhR"],["hhR","khR"],["khR","ahR"]];

function NodeSkeleton({ c, species }) {
  const pr = useRef();
  const n = species === "rat" ? R_NODES : H_NODES;
  const l = species === "rat" ? R_LINKS : H_LINKS;
  const pa = useMemo(() => new Float32Array(Object.values(n).flat()), [n]);
  const la = useMemo(() => { const o = []; l.forEach(([a, b]) => o.push(...n[a], ...n[b])); return new Float32Array(o); }, [l, n]);
  useFrame(({ clock }) => { if (pr.current) pr.current.material.opacity = 0.52 + Math.sin(clock.elapsedTime * 2.2) * 0.13; });
  return <><lineSegments><bufferGeometry><bufferAttribute attach="attributes-position" count={la.length / 3} array={la} itemSize={3} /></bufferGeometry><lineBasicMaterial color={c} transparent opacity={0.22} /></lineSegments><points ref={pr}><bufferGeometry><bufferAttribute attach="attributes-position" count={pa.length / 3} array={pa} itemSize={3} /></bufferGeometry><pointsMaterial color={c} size={species === "rat" ? 0.03 : 0.04} transparent opacity={0.56} /></points></>;
}

const H_SPOTS = { brain:{color:"#00E5FF",pos:[0,2.02,0]}, heart:{color:"#FF1A3C",pos:[-0.14,1.08,0.2]}, liver:{color:"#FF6B35",pos:[0.2,0.62,0.18]}, lungs:{color:"#88CCFF",pos:[0,0.98,0.16]}, spine:{color:"#FFD700",pos:[0,0.5,-0.07]}, dna:{color:"#FF3EAA",pos:[0.36,-0.1,0.12]} };
const R_SPOTS = { brain:{color:"#00E5FF",pos:[1.05,0.43,0]}, heart:{color:"#FF1A3C",pos:[0.38,0.25,0.05]}, liver:{color:"#FF6B35",pos:[0.08,0.16,0.08]}, lungs:{color:"#88CCFF",pos:[0.34,0.3,0]}, spine:{color:"#FFD700",pos:[0.02,0.29,0]}, dna:{color:"#FF3EAA",pos:[-0.56,0.22,0.1]} };

function BodyScene({ species = "human", highlight = null }) {
  const isRat = species === "rat";
  const c = isRat ? "#39FF88" : "#00E5FF";
  const spots = isRat ? R_SPOTS : H_SPOTS;
  const gr = useRef();
  const oc = (id) => highlight === id ? spots[id].color : c;
  const oo = (id) => highlight === id ? 0.95 : 0.65;
  const h = useMemo(() => ({ head: ellipsoid(0.3,0.4,0.29,52), torso: ellipsoid(0.56,0.9,0.38,64), pelvis: ellipsoid(0.35,0.24,0.28,44), spine: tube([[0,1.62,-0.05],[0,1.3,-0.06],[0,0.8,-0.07],[0,0.2,-0.06],[0,-0.55,-0.02]],0.04,8), armU: tube([[0,0,0],[0,-0.35,0.03],[0,-0.6,0.04]],0.07,8), armL: tube([[0,0,0],[0,-0.28,0.01],[0,-0.52,0.01]],0.055,8), legU: tube([[0,0,0],[0.02,-0.5,0.03],[0,-0.85,0.03]],0.09,8), legL: tube([[0,0,0],[0,-0.42,0.03],[0,-0.75,0.04]],0.07,8), lung: ellipsoid(0.16,0.28,0.14,34), heart: ellipsoid(0.12,0.16,0.11,36), liver: ellipsoid(0.2,0.1,0.14,30), brain: ellipsoid(0.14,0.1,0.17,32) }), []);
  const r = useMemo(() => ({ head: ellipsoid(0.22,0.18,0.24,46), body: ellipsoid(0.94,0.24,0.32,60), pelvis: ellipsoid(0.36,0.18,0.24,40), spine: tube([[0.9,0.34,0],[0.54,0.32,0],[0.14,0.3,0],[-0.24,0.27,0],[-0.62,0.24,0],[-0.92,0.2,0]],0.028,7), tail: tube([[-0.92,0.2,0],[-1.2,0.18,0],[-1.45,0.14,0.05],[-1.7,0.07,0.1]],0.02,7), neck: tube([[0,0,0],[0.08,0.04,0],[0.16,0.05,0]],0.045,7), fU: tube([[0,0,0],[-0.09,-0.11,0.01],[-0.13,-0.2,0.03]],0.034,7), fL: tube([[0,0,0],[-0.07,-0.13,0.02],[-0.08,-0.2,0.06]],0.026,7), hU: tube([[0,0,0],[-0.12,-0.1,0.02],[-0.17,-0.2,0.05]],0.04,7), hL: tube([[0,0,0],[-0.07,-0.13,0.04],[-0.03,-0.22,0.08]],0.03,7), paw: tube([[0,0,0],[0.02,-0.01,0.05],[0.06,-0.01,0.1]],0.018,6), lung: ellipsoid(0.12,0.12,0.09,26), heart: ellipsoid(0.09,0.1,0.08,26), liver: ellipsoid(0.18,0.09,0.13,26), brain: ellipsoid(0.12,0.08,0.16,26) }), []);
  useFrame(({ clock }) => { if (!gr.current) return; const t = clock.elapsedTime; gr.current.rotation.y = Math.sin(t * 0.08) * (isRat ? 0.14 : 0.1); if (isRat) gr.current.rotation.x = Math.sin(t * 0.09) * 0.02; });
  if (isRat) return <group ref={gr} position={[0,-0.15,0]} scale={1.45}><ambientLight intensity={0.12} /><pointLight position={[3,3,3]} intensity={2.2} color={c} /><group position={[1.03,0.42,0]}><Holo g={r.head} c={c} o={0.34} /><Holo g={r.head} c={c} o={0.09} w /><group position={[0.02,0.02,0]}><Holo g={r.brain} c={oc("brain")} o={oo("brain")} /></group></group><group position={[0.84,0.34,0]}><Holo g={r.neck} c={c} o={0.42} /></group><group position={[0.1,0.28,0]}><Holo g={r.body} c={c} o={0.2} /></group><group position={[-0.72,0.22,0]}><Holo g={r.pelvis} c={c} o={0.36} /></group><Holo g={r.spine} c={oc("spine")} o={0.86} /><Holo g={r.tail} c={c} o={0.58} /><group position={[0.34,0.3,0.09]}><Holo g={r.lung} c={oc("lungs")} o={0.58} /></group><group position={[0.34,0.3,-0.09]}><Holo g={r.lung} c={oc("lungs")} o={0.58} /></group><group position={[0.38,0.25,0.05]}><Holo g={r.heart} c={oc("heart")} o={oo("heart")} /></group><group position={[0.08,0.16,0.08]}><Holo g={r.liver} c={oc("liver")} o={oo("liver")} /></group>{[1,-1].map((s,i)=><group key={`rf-${i}`} position={[0.44,0.22,s*0.16]} rotation={[0.1,s*0.06,0.08]}><Holo g={r.fU} c={c} o={0.52} /><group position={[-0.12,-0.2,0.03]}><Holo g={r.fL} c={c} o={0.5} /><group position={[-0.07,-0.2,0.06]}><Holo g={r.paw} c={c} o={0.5} /></group></group></group>)}{[1,-1].map((s,i)=><group key={`rh-${i}`} position={[-0.58,0.2,s*0.16]} rotation={[0.08,s*0.06,-0.05]}><Holo g={r.hU} c={c} o={0.55} /><group position={[-0.17,-0.2,0.05]}><Holo g={r.hL} c={c} o={0.5} /><group position={[-0.03,-0.22,0.08]}><Holo g={r.paw} c={c} o={0.5} /></group></group></group>)}<NodeSkeleton c={c} species="rat" /></group>;
  return <group ref={gr} position={[0,-1.8,0]} scale={0.88}><ambientLight intensity={0.11} /><pointLight position={[3,6,4]} intensity={3} color={c} /><group position={[0,2.05,0]}><Holo g={h.head} c={c} o={0.34} /><Holo g={h.head} c={c} o={0.09} w /><group position={[0,0.02,0]}><Holo g={h.brain} c={oc("brain")} o={oo("brain")} /></group></group><group position={[0,0.55,0]}><Holo g={h.torso} c={c} o={0.18} /></group><group position={[0,-0.55,0]}><Holo g={h.pelvis} c={c} o={0.34} /></group><Holo g={h.spine} c={oc("spine")} o={0.84} /><group position={[0.26,0.98,0.1]}><Holo g={h.lung} c={oc("lungs")} o={0.55} /></group><group position={[-0.26,0.98,0.1]}><Holo g={h.lung} c={oc("lungs")} o={0.55} /></group><group position={[-0.14,1.08,0.2]}><Holo g={h.heart} c={oc("heart")} o={oo("heart")} /></group><group position={[0.2,0.62,0.18]}><Holo g={h.liver} c={oc("liver")} o={oo("liver")} /></group>{[1,-1].map((s,i)=><group key={`ha-${i}`} position={[s*0.72,1.2,0.05]} rotation={[0.08,0,s*0.18]}><Holo g={h.armU} c={c} o={0.48} /><group position={[0,-0.62,0.03]}><Holo g={h.armL} c={c} o={0.46} /></group></group>)}{[0.2,-0.2].map((x,i)=><group key={`hl-${i}`} position={[x,-0.55,0.02]}><Holo g={h.legU} c={c} o={0.49} /><group position={[0,-0.9,0.02]}><Holo g={h.legL} c={c} o={0.47} /></group></group>)}<NodeSkeleton c={c} species="human" /></group>;
}

function Rings({ c }) { const a = useRef(); const b = useRef(); useFrame(({ clock }) => { const t = clock.elapsedTime; if (a.current) a.current.rotation.x = t * 0.34; if (b.current) b.current.rotation.y = t * 0.24; }); return <><Torus ref={a} args={[1.85,0.008,8,84]}><meshStandardMaterial color={c} emissive={c} emissiveIntensity={1} transparent opacity={0.35} /></Torus><Torus ref={b} args={[2.06,0.007,8,84]}><meshStandardMaterial color={c} emissive={c} emissiveIntensity={1} transparent opacity={0.24} /></Torus></>; }
function Toggle({ c, mode, setMode, a, b, top = 68 }) { const h = mode === "human"; return <div style={{position:"absolute",top,left:"50%",transform:"translateX(-50%)",zIndex:30,display:"flex",fontFamily:"'Orbitron', monospace"}}><button onClick={()=>setMode("human")} style={{padding:"9px 22px",background:h?c:"transparent",border:`1px solid ${h?c:`${c}44`}`,color:h?"#041116":`${c}88`,borderRight:"none",fontSize:10,letterSpacing:2}}>{a}</button><button onClick={()=>setMode("rat")} style={{padding:"9px 22px",background:!h?c:"transparent",border:`1px solid ${!h?c:`${c}44`}`,color:!h?"#041116":`${c}88`,fontSize:10,letterSpacing:2}}>{b}</button></div>; }

function HomePage({ onNav }) {
  const [mode, setMode] = useState("human");
  const [hov, setHov] = useState(null);
  const [tm, setTm] = useState(new Date());
  useEffect(()=>{const t=setInterval(()=>setTm(new Date()),1000); return ()=>clearInterval(t);},[]);
  const rat = mode === "rat";
  const c = rat ? "#39FF88" : "#00E5FF";
  const cards = [
    { id:"brain", route:"neural", label:"NEURAL ATLAS", sub:"Brain Anatomy", color:"#00E5FF", stat: rat ? "71M neurons" : "86B neurons" },
    { id:"heart", route:"heart", label:"CARDIAC ATLAS", sub:"Heart Anatomy", color:"#FF1A3C", stat: rat ? "250-450 bpm" : "5 L/min output" },
    { id:"liver", route:"liver", label:"HEPATIC ATLAS", sub:"Liver Anatomy", color:"#FF6B35", stat: rat ? "6 lobes" : "500+ functions" },
    { id:"dna", route:"genomic", label:"GENOMIC ATLAS", sub:"DNA Structure", color:"#FF3EAA", stat: rat ? "2.75B base pairs" : "3.2B base pairs" },
  ];
  return <div style={{width:"100%",height:"100%",position:"relative",overflow:"hidden"}}><div style={{position:"absolute",left:0,right:0,height:1,background:`linear-gradient(to right,transparent,${c}44 30%,${c}44 70%,transparent)`,animation:"scanLine 7s linear infinite",zIndex:2}} /><div style={{position:"absolute",left:"50%",top:0,bottom:32,width:"min(620px, calc(100% - 520px))",minWidth:320,transform:"translateX(-50%)",zIndex:5}}><Canvas camera={{position:[0,0.5,6.5],fov:44}} gl={{antialias:true,alpha:true}} style={{background:"transparent",width:"100%",height:"100%"}}><Suspense fallback={null}><BodyScene highlight={hov} species={mode} /><OrbitControls enablePan={false} minDistance={3.5} maxDistance={11} /></Suspense></Canvas></div><Toggle c={c} mode={mode} setMode={setMode} a="HUMAN BODY" b="RAT BODY" top={22} /><div style={{position:"absolute",top:78,left:"50%",transform:"translateX(-50%)",zIndex:20,textAlign:"center",fontFamily:"'Orbitron', monospace",pointerEvents:"none"}}><div style={{color:c,fontSize:9,letterSpacing:5,opacity:0.5}}>HOLOGRAPHIC ANATOMY SYSTEM</div><div style={{color:c,fontSize:22,letterSpacing:7,fontWeight:900,textShadow:`0 0 35px ${c},0 0 70px ${c}44`}}>{rat?"RAT ATLAS":"HUMAN ATLAS"}</div></div><div style={{position:"absolute",left:18,top:116,bottom:40,width:220,zIndex:10,fontFamily:"'Orbitron', monospace"}}><div style={{color:c,fontSize:8,letterSpacing:3,opacity:0.5,marginBottom:8}}>SPECIMEN</div><div style={{padding:"8px 10px",border:`1px solid ${c}18`,borderRadius:3,background:`${c}04`,marginBottom:8}}><div style={{color:c,fontSize:11,letterSpacing:2,fontWeight:700}}>{rat?"RATTUS NORVEGICUS":"HOMO SAPIENS"}</div><div style={{color:`${c}55`,fontSize:7.5,letterSpacing:1.3,marginTop:2}}>{rat?"ADULT · 0.35KG · 23CM · 12WKS":"ADULT · 70KG · 175CM · 35YRS"}</div></div><div style={{color:`${c}33`,fontSize:7.5,fontFamily:"'Share Tech Mono', monospace"}}>{tm.toLocaleString()}</div></div><div style={{position:"absolute",right:18,top:116,bottom:40,width:235,zIndex:10,display:"flex",flexDirection:"column",gap:10,fontFamily:"'Orbitron', monospace"}}>{cards.map((x)=><button key={x.id} onClick={()=>onNav(x.route)} onMouseEnter={()=>setHov(x.id)} onMouseLeave={()=>setHov(null)} style={{padding:"10px 12px",background:hov===x.id?`${x.color}14`:`${c}02`,border:`1px solid ${hov===x.id?x.color:`${x.color}33`}`,borderRadius:4,color:x.color,textAlign:"left"}}><div style={{fontSize:9,letterSpacing:2,fontWeight:700}}>{x.label}</div><div style={{fontSize:8,opacity:0.75}}>{x.sub}</div><div style={{fontSize:8,opacity:0.75,marginTop:4}}>{x.stat}</div></button>)}</div></div>;
}

function BrainSpin({ g, c }) {
  const r = useRef();
  useFrame(({ clock }) => {
    if (r.current) r.current.rotation.y = clock.elapsedTime * 0.12;
  });
  return <group ref={r}><Holo g={g} c={c} o={0.9} /><Holo g={g} c={c} o={0.16} w /></group>;
}

function BrainPage(){
  const [mode,setMode]=useState("human");
  const rat=mode==="rat";
  const c=rat?"#39FF88":"#00E5FF";
  const g=useMemo(()=>buildBrain(rat),[rat]);
  return <div style={{width:"100%",height:"100%",position:"relative",background:"#020810"}}>
    <Toggle c={c} mode={mode} setMode={setMode} a="HUMAN BRAIN" b="RAT BRAIN" />
    <Canvas camera={{position:[0,0,4],fov:50}} style={{width:"100%",height:"100%"}}>
      <ambientLight intensity={0.15} />
      <pointLight position={[4,4,4]} intensity={1.5} color={c} />
      <BrainSpin g={g} c={c} />
      <Rings c={c} />
      <OrbitControls enablePan={false} minDistance={2.5} maxDistance={6.5} />
    </Canvas>
  </div>;
}

function OrganSpin({ g, color }) {
  const r = useRef();
  useFrame(({ clock }) => {
    if (r.current) r.current.rotation.y = clock.elapsedTime * 0.12;
  });
  return <group ref={r}><Holo g={g} c={color} o={0.9} /><Holo g={g} c={color} o={0.16} w /></group>;
}

function OrganPage({ color, label, geoFn }){
  const [mode,setMode]=useState("human");
  const rat=mode==="rat";
  const g=useMemo(()=>geoFn(rat),[rat,geoFn]);
  return <div style={{width:"100%",height:"100%",position:"relative",background:"#030710"}}>
    <Toggle c={color} mode={mode} setMode={setMode} a={`HUMAN ${label}`} b={`RAT ${label}`} />
    <Canvas camera={{position:[0,0,4.5],fov:50}} style={{width:"100%",height:"100%"}}>
      <ambientLight intensity={0.12} />
      <pointLight position={[4,4,4]} intensity={1.8} color={color} />
      <OrganSpin g={g} color={color} />
      <Rings c={color} />
      <OrbitControls enablePan={false} minDistance={2.8} maxDistance={8} />
    </Canvas>
  </div>;
}

function DNAPage(){ return <OrganPage color="#7abfff" label="DNA" geoFn={(rat)=>ellipsoid(rat?0.95:1.08,rat?0.95:1.08,rat?0.95:1.08,72)} />; }
function HeartPage(){ return <OrganPage color="#FF1A3C" label="HEART" geoFn={(rat)=>ellipsoid(rat?0.58:0.85,rat?0.7:1,rat?0.52:0.76,64)} />; }
function LiverPage(){ return <OrganPage color="#FF6B35" label="LIVER" geoFn={(rat)=>ellipsoid(rat?0.78:1.05,rat?0.42:0.56,rat?0.62:0.76,64)} />; }

const NAV=[{id:"home",label:"ATLAS",color:"#00E5FF"},{id:"brain",label:"NEURAL",color:"#00E5FF"},{id:"heart",label:"CARDIAC",color:"#FF1A3C"},{id:"liver",label:"HEPATIC",color:"#FF6B35"},{id:"dna",label:"GENOMIC",color:"#FF3EAA"}];
function NavBar({page,onNav}){ const [tm,setTm]=useState(new Date()); useEffect(()=>{const t=setInterval(()=>setTm(new Date()),1000); return ()=>clearInterval(t);},[]); return <div style={{position:"fixed",top:0,left:0,right:0,height:58,zIndex:1000,display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 22px",background:"rgba(2,8,16,0.96)",borderBottom:"1px solid rgba(0,229,255,0.1)",fontFamily:"'Orbitron', monospace"}}><div style={{color:"#00E5FF",fontSize:12,letterSpacing:4,fontWeight:700}}>BIOATLAS</div><div style={{display:"flex",gap:6}}>{NAV.map((n)=>{const a=page===n.id; return <button key={n.id} onClick={()=>onNav(n.id)} style={{padding:"6px 15px",background:a?`${n.color}18`:"transparent",border:`1px solid ${a?n.color:`${n.color}33`}`,color:a?n.color:`${n.color}77`,fontSize:9,letterSpacing:2.3}}>{n.label}</button>;})}</div><div style={{color:"rgba(0,229,255,0.35)",fontSize:8,fontFamily:"'Share Tech Mono', monospace"}}>{tm.toLocaleTimeString()}</div></div>; }

export default function BioLabApp(){
  const [page,setPage]=useState("home");
  const norm=(p)=>p==="neural"?"brain":p==="genomic"?"dna":p;
  const go=(p)=>setPage(norm(p));
  const pages={home:<HomePage onNav={go} />,brain:<BrainPage />,heart:<HeartPage />,liver:<LiverPage />,dna:<DNAPage />};
  return <div style={{width:"100vw",height:"100vh",background:"#020810",overflow:"hidden",position:"relative"}}><div style={{position:"absolute",inset:0,opacity:0.055,backgroundImage:"linear-gradient(rgba(0,229,255,0.06) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,0.06) 1px,transparent 1px)",backgroundSize:"42px 42px"}} /><NavBar page={page} onNav={go} /><div style={{position:"absolute",inset:0,top:58}}>{pages[page]}</div><style>{`@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Share+Tech+Mono&display=swap');*{box-sizing:border-box;margin:0;padding:0;}button{font-family:inherit;cursor:crosshair;border-radius:3px;}@keyframes scanLine{from{transform:translateY(-100%)}to{transform:translateY(100vh)}}`}</style></div>;
}
