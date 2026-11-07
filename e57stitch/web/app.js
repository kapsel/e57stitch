import * as THREE from 'three';

/* ------------------------------------------------------------------ utils */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}
async function postJSON(url, body, method = 'POST') {
  const r = await fetch(url, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmtInt = (n) => (n == null ? '—' : n.toLocaleString());
const fmtPts = (n) => {
  if (n == null) return '—';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(0) + 'k';
  return String(n);
};
const fmtSize = (n) => {
  if (!n) return '0 B';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(i ? 1 : 0)} ${u[i]}`;
};
const fmtM = (v) => (Math.abs(v) < 10 ? v.toFixed(3) : v.toFixed(2));
const rotDeg = (rad) => (rad * 180) / Math.PI;
const rotRad = (deg) => (deg * Math.PI) / 180;

function mat4FromRow(a) {
  const m = new THREE.Matrix4();
  m.fromArray(a.flat());
  m.transpose(); // server matrices are row-major
  return m;
}

function hexRGB(hex) {
  const h = hex.replace('#', '');
  return {
    r: parseInt(h.slice(0, 2), 16) / 255,
    g: parseInt(h.slice(2, 4), 16) / 255,
    b: parseInt(h.slice(4, 6), 16) / 255,
  };
}

function toast(msg, kind = '', action = null) {
  const el = $('#toast');
  el.innerHTML = `<span>${escapeHtml(msg)}</span>` + (action ? ` <button class="toast-action">${escapeHtml(action.label)}</button>` : '');
  el.className = `toast ${kind}`;
  if (action) {
    el.querySelector('.toast-action').onclick = () => {
      el.classList.add('hidden');
      action.fn();
    };
  }
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.add('hidden'), action ? 12000 : 4200);
}

/* ------------------------------------------------------------------ state */

const app = {
  state: null,
  scans: new Map(), // sid -> {data, points, material, fetching, offset, featureObj}
  selected: null,
  layout: 'grid',
  layoutUserSet: false,
  colorMode: 'scan',
  showFeatures: false,
  showBefore: false,
  before: new Map(),
  spread: 0,
  job: null,
  lastHash: '',
  reloadPending: false,
  fitPending: true,
  activeGroup: 0,
  pairFilter: 'all',
  hidden: new Set(),
  mainOnly: false,
  showScannerLinks: false,
  focusedGroup: null,
  paramsDirty: false,
  selection: new Set(),
  transformUndo: null,
  lastMeshPath: null,
  fly: false,
  flyPos: new THREE.Vector3(),
  flyYaw: 0,
  flyPitch: 0,
  flyKeys: new Set(),
  flyLast: 0,
  cleanupMode: false,
  cleanupDraft: null,
  clipDraft: null, // local edits of the clip box, not yet confirmed by the server
  clipSaveTimer: null,
};

/* ------------------------------------------------------------------ three */

const canvas = $('#view');
const wrap = $('#viewport-wrap');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x0a0c12, 1);
renderer.setScissorTest(true);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0a0c12);
const root = new THREE.Group();
root.matrixAutoUpdate = false;
scene.add(root);

const gridGroup = new THREE.Group();
gridGroup.matrixAutoUpdate = false;
scene.add(gridGroup);

const meshGroup = new THREE.Group();
scene.add(meshGroup);
const cleanupGroup = new THREE.Group();
scene.add(cleanupGroup);
scene.add(new THREE.AmbientLight(0xffffff, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.35);
sun.position.set(-40, -60, 80);
scene.add(sun);

const topCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1e7);
const sideCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1e7);
const threeCam = new THREE.PerspectiveCamera(42, 1, 0.05, 1e7);
threeCam.up.set(0, 0, 1);

const views = {
  top: { camera: topCam, center: new THREE.Vector3(), half: 10, distance: 200, normal: new THREE.Vector3(0, 0, 1) },
  side: { camera: sideCam, center: new THREE.Vector3(), half: 10, distance: 200, normal: new THREE.Vector3(0, -1, 0) },
};
const orbit = {
  target: new THREE.Vector3(),
  radius: 30,
  theta: -Math.PI / 3,
  phi: Math.PI / 3.1,
};

const panes = []; // {view, x, y, w, h} in WebGL coordinates
const panesCSS = []; // {view, x, y, w, h} in CSS coordinates

function layoutViewports() {
  const w = Math.max(2, wrap.clientWidth);
  const h = Math.max(2, wrap.clientHeight);
  renderer.setSize(w, h, false);
  const topH = Math.round(h * 0.55);
  const halfW = Math.floor(w / 2);
  panesCSS.length = 0;
  panesCSS.push({ view: 'top', x: 0, y: 0, w: halfW, h: topH });
  panesCSS.push({ view: 'side', x: halfW, y: 0, w: w - halfW, h: topH });
  panesCSS.push({ view: 'three', x: 0, y: topH, w, h: h - topH });
  panes.length = 0;
  for (const p of panesCSS) panes.push({ ...p, y: h - (p.y + p.h) });
  requestRender();
}

window.addEventListener('resize', layoutViewports);

/* ------------------------------------------------------------------ shaders */

const VERT = /* glsl */ `
  attribute vec3 aColor;
  attribute float aIntensity;
  uniform float uSize;
  uniform bool uOrtho;
  uniform int uColorMode;
  uniform vec3 uColor;
  uniform float uHeightMin;
  uniform float uHeightMax;
  uniform float uDim;
  uniform mat4 uToReg;
  varying vec3 vColor;
  varying vec3 vWorld;
  varying vec3 vReg;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vec4 world = modelMatrix * vec4(position, 1.0);
    vReg = (uToReg * world).xyz;
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uOrtho ? uSize : clamp(uSize * (260.0 / max(-mv.z, 0.05)), 1.0, 7.0);
    vec3 c;
    if (uColorMode == 1) {
      c = aColor;
    } else if (uColorMode == 2) {
      float t = clamp((world.z - uHeightMin) / max(uHeightMax - uHeightMin, 1e-6), 0.0, 1.0);
      c = mix(vec3(0.02, 0.18, 0.55), vec3(1.0, 0.78, 0.15), t);
      c = mix(c, vec3(0.95, 0.28, 0.28), smoothstep(0.75, 1.0, t));
    } else if (uColorMode == 3) {
      c = vec3(aIntensity);
    } else {
      c = uColor;
    }
    vColor = c * uDim;
    vWorld = world.xyz;
  }
`;

const FRAG = /* glsl */ `
  uniform int uCleanCount;
  uniform vec4 uCleanC[32];
  uniform vec4 uCleanS[32];
  uniform bool uClipOn;
  uniform vec3 uClipMin;
  uniform vec3 uClipMax;
  varying vec3 vColor;
  varying vec3 vWorld;
  varying vec3 vReg;
  void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;
    if (uClipOn && (any(lessThan(vReg, uClipMin)) || any(greaterThan(vReg, uClipMax)))) discard;
    for (int i = 0; i < 32; i++) {
      if (i >= uCleanCount) break;
      vec4 c = uCleanC[i];
      vec4 s = uCleanS[i];
      if (s.w > 0.5) {
        if (distance(vWorld, c.xyz) <= s.x) discard;
      } else {
        vec3 dd = abs(vWorld - c.xyz);
        if (all(lessThanEqual(dd, s.xyz))) discard;
      }
    }
    gl_FragColor = vec4(vColor, 1.0);
  }
`;

function clipUniforms() {
  return {
    uToReg: { value: new THREE.Matrix4() },
    uClipOn: { value: false },
    uClipMin: { value: new THREE.Vector3(-1e9, -1e9, -1e9) },
    uClipMax: { value: new THREE.Vector3(1e9, 1e9, 1e9) },
  };
}

function makeMaterial(scan, colorHex) {
  const col = hexRGB(colorHex);
  return new THREE.ShaderMaterial({
    uniforms: {
      uSize: { value: 1.7 },
      uOrtho: { value: true },
      uColorMode: { value: 0 },
      uColor: { value: new THREE.Vector3(col.r, col.g, col.b) },
      uHeightMin: { value: 0 },
      uHeightMax: { value: 10 },
      uDim: { value: 1 },
      uCleanCount: { value: 0 },
      uCleanC: { value: Array.from({ length: 32 }, () => new THREE.Vector4()) },
      uCleanS: { value: Array.from({ length: 32 }, () => new THREE.Vector4()) },
      ...clipUniforms(),
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    transparent: false,
    depthTest: true,
    depthWrite: true,
  });
}

const featureMaterialEdge = new THREE.ShaderMaterial({
  uniforms: {
    uSize: { value: 3.4 },
    uOrtho: { value: true },
    uColorMode: { value: 0 },
    uColor: { value: new THREE.Vector3(1.0, 0.12, 0.72) },
    uHeightMin: { value: 0 },
    uHeightMax: { value: 1 },
    uDim: { value: 1 },
    uCleanCount: { value: 0 },
    uCleanC: { value: Array.from({ length: 32 }, () => new THREE.Vector4()) },
    uCleanS: { value: Array.from({ length: 32 }, () => new THREE.Vector4()) },
    ...clipUniforms(),
  },
  vertexShader: VERT,
  fragmentShader: FRAG,
  transparent: false,
  depthTest: true,
  depthWrite: false,
});
const featureMaterialPlane = featureMaterialEdge.clone();
featureMaterialPlane.uniforms = THREE.UniformsUtils.clone(featureMaterialEdge.uniforms);
featureMaterialPlane.uniforms.uColor.value = new THREE.Vector3(1.0, 0.75, 0.05);
featureMaterialPlane.uniforms.uSize.value = 2.6;

/* ------------------------------------------------------------------ render */

let needsRender = true;
function requestRender() { needsRender = true; }

function updateOrtho(view, pane) {
  const cam = view.camera;
  const aspect = pane.w / pane.h;
  let halfH = view.half;
  if (aspect < 1) halfH = view.half / aspect;
  const halfW = halfH * aspect;
  cam.left = -halfW;
  cam.right = halfW;
  cam.top = halfH;
  cam.bottom = -halfH;
  cam.near = Math.max(0.05, view.distance * 0.001);
  cam.far = view.distance * 3;
  const c = view.center;
  cam.position.copy(c).addScaledVector(view.normal, view.distance);
  if (view === views.top) {
    cam.up.set(0, 1, 0);
    cam.lookAt(c.x, c.y, c.z);
  } else {
    cam.up.set(0, 0, 1);
    cam.lookAt(c.x, c.y, c.z);
  }
  cam.updateProjectionMatrix();
}

function updateThreeCamera(pane) {
  threeCam.aspect = pane.w / pane.h;
  threeCam.updateProjectionMatrix();
  if (app.fly) {
    const cp = Math.cos(app.flyPitch);
    const dir = new THREE.Vector3(
      cp * Math.cos(app.flyYaw),
      cp * Math.sin(app.flyYaw),
      Math.sin(app.flyPitch)
    );
    threeCam.position.copy(app.flyPos);
    threeCam.up.set(0, 0, 1);
    threeCam.lookAt(app.flyPos.clone().add(dir));
    return;
  }
  const sp = Math.sin(orbit.phi);
  threeCam.position.set(
    orbit.target.x + orbit.radius * sp * Math.cos(orbit.theta),
    orbit.target.y + orbit.radius * sp * Math.sin(orbit.theta),
    orbit.target.z + orbit.radius * Math.cos(orbit.phi)
  );
  threeCam.up.set(0, 0, 1);
  threeCam.lookAt(orbit.target);
}

const _bbox = new THREE.Box3();
function currentHeightRange() {
  const bb = robustBBox();
  if (!bb) return [0, 10];
  return [bb.min.z, bb.max.z];
}

function applyPointUniforms(ortho) {
  const [zmin, zmax] = currentHeightRange();
  const mode = app.colorMode === 'rgb' ? 1 : app.colorMode === 'height' ? 2 : 0;
  for (const s of app.scans.values()) {
    if (!s.material) continue;
    const u = s.material.uniforms;
    u.uOrtho.value = ortho;
    u.uSize.value = (ortho ? 1.8 : 2.4) * renderer.getPixelRatio();
    u.uColorMode.value = mode;
    u.uHeightMin.value = zmin;
    u.uHeightMax.value = zmax;
    const regions = (app.state?.cleanups || []).filter(
      (c) => c.enabled !== false && c.op !== 'keep' && !c.clip && (!c.scans || c.scans.includes(s.data.id))
    ).slice(0, 32);
    const clip = activeClip();
    u.uClipOn.value = !!clip;
    if (clip) {
      u.uClipMin.value.set(clip.min[0], clip.min[1], clip.min[2]);
      u.uClipMax.value.set(clip.max[0], clip.max[1], clip.max[2]);
      // display position -> registration world (what export uses): T * inv(display)
      const T = mat4FromRow(s.data.transform);
      u.uToReg.value.copy(T.multiply(scanDisplayMatrix(s).invert()));
    }
    u.uCleanCount.value = regions.length;
    for (let i = 0; i < regions.length; i++) {
      const c = regions[i];
      u.uCleanC.value[i].set(c.center[0], c.center[1], c.center[2], 1);
      if (c.kind === 'sphere') u.uCleanS.value[i].set(c.radius || 1, 0, 0, 1);
      else u.uCleanS.value[i].set(c.size?.[0] || 1, c.size?.[1] || 1, c.size?.[2] || 1, 0);
    }
    const A = app.assist;
    if (A) {
      const moving = A.movingSet.has(s.data.id);
      const target = A.targetIds.includes(s.data.id);
      const hex = moving ? ASSIST_MOVING : ASSIST_TARGET;
      const c = hexRGB(hex);
      u.uColor.value.set(c.r, c.g, c.b);
      u.uColorMode.value = A.colors === 'rgb' ? 1 : 0;
      u.uDim.value = moving || A.colors === 'rgb' ? 1.0 : 0.8;
      s.points.visible = moving || target;
      if (s.featureEdge) s.featureEdge.visible = s.points.visible && app.showFeatures;
      if (s.featurePlane) s.featurePlane.visible = s.points.visible && app.showFeatures;
      continue;
    }
    {
      const c = hexRGB(s.data.color);
      u.uColor.value.set(c.r, c.g, c.b);
    }
    let dim = 1.0;
    if (app.selected && app.selected !== s.data.id) dim = 0.55;
    if (app.highlightPair) {
      const [a, b] = app.highlightPair.split('|');
      if (s.data.id !== a && s.data.id !== b) dim = Math.min(dim, 0.4);
    }
    // let the scanner links stand out, unless a scan or link is being looked at
    if (app.showScannerLinks && !app.selected && !app.highlightPair) dim = 0.35;
    const inMain = !app.mainOnly || s.data.group === 0 || s.data.group === -1;
    s.points.visible = !app.hidden.has(s.data.id) && inMain;
    if (s.featureEdge) s.featureEdge.visible = s.points.visible && app.showFeatures;
    if (s.featurePlane) s.featurePlane.visible = s.points.visible && app.showFeatures;
    u.uDim.value = dim;
  }
  // feature overlays share one material across scans: clip them in display space
  const fclip = activeClip();
  for (const m of [featureMaterialEdge, featureMaterialPlane]) {
    m.uniforms.uClipOn.value = !!fclip;
    if (fclip) {
      m.uniforms.uClipMin.value.set(fclip.min[0], fclip.min[1], fclip.min[2]);
      m.uniforms.uClipMax.value.set(fclip.max[0], fclip.max[1], fclip.max[2]);
    }
  }
  featureMaterialEdge.uniforms.uOrtho.value = ortho;
  featureMaterialPlane.uniforms.uOrtho.value = ortho;
  featureMaterialEdge.uniforms.uSize.value = (ortho ? 3.4 : 4.0) * renderer.getPixelRatio();
  featureMaterialPlane.uniforms.uSize.value = (ortho ? 2.6 : 3.2) * renderer.getPixelRatio();
}

function render() {
  if (!needsRender) return;
  needsRender = false;
  updateScannerLinks();
  for (const pane of panes) {
    const view = pane.view === 'three' ? null : views[pane.view];
    if (view) updateOrtho(view, pane);
    else updateThreeCamera(pane);
    applyPointUniforms(pane.view !== 'three');
    renderer.setViewport(pane.x, pane.y, pane.w, pane.h);
    scannerLineMaterial.uniforms.uViewport.value.set(pane.w, pane.h);
    renderer.setScissor(pane.x, pane.y, pane.w, pane.h);
    renderer.render(scene, view ? view.camera : threeCam);
  }
}

function flyStep() {
  if (!app.fly) return;
  const now = performance.now();
  const dt = Math.min(0.06, (now - (app.flyLast || now)) / 1000);
  app.flyLast = now;
  const keys = app.flyKeys;
  const speed = (keys.has('shift') ? 9.0 : 3.0) * dt;
  const cp = Math.cos(app.flyPitch);
  const forward = new THREE.Vector3(
    cp * Math.cos(app.flyYaw),
    cp * Math.sin(app.flyYaw),
    Math.sin(app.flyPitch)
  );
  const right = new THREE.Vector3(-Math.sin(app.flyYaw), Math.cos(app.flyYaw), 0);
  if (keys.has('w')) app.flyPos.addScaledVector(forward, speed);
  if (keys.has('s')) app.flyPos.addScaledVector(forward, -speed);
  if (keys.has('d')) app.flyPos.addScaledVector(right, speed);
  if (keys.has('a')) app.flyPos.addScaledVector(right, -speed);
  if (keys.has('e')) app.flyPos.z += speed;
  if (keys.has('q')) app.flyPos.z -= speed;
}

function animate() {
  requestAnimationFrame(animate);
  if (app.fly) {
    flyStep();
    requestRender();
  }
  render();
}
requestAnimationFrame(animate);

/* ------------------------------------------------------------------ layout */

function scanDisplayMatrix(scan) {
  if (app.assist && app.assist.movingSet.has(scan.data.id)) return assistDisplayMatrix(scan.data.id);
  const t = mat4FromRow(
    app.showBefore && app.before.has(scan.data.id) ? app.before.get(scan.data.id) : scan.data.transform
  );
  if (app.layout === 'grid') {
    return scan.offset.clone();
  }
  return scan.offset.clone().multiply(t);
}

function worldBBox(members = null) {
  let bb = null;
  for (const s of app.scans.values()) {
    if (members && !members.includes(s.data.id)) continue;
    if (app.hidden.has(s.data.id)) continue;
    if (app.mainOnly && s.data.group !== 0 && s.data.group !== -1) continue;
    if (!s.data.loaded || !s.data.bbox) continue;
    const local = s.data.bbox;
    const M = scanDisplayMatrix(s);
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(
        i & 1 ? local[1][0] : local[0][0],
        i & 2 ? local[1][1] : local[0][1],
        i & 4 ? local[1][2] : local[0][2]
      ).applyMatrix4(M);
      if (!bb) bb = new THREE.Box3(p.clone(), p.clone());
      else bb.expandByPoint(p);
    }
  }
  return bb;
}

function robustBBox(members = null) {
  const xs = [];
  const ys = [];
  const zs = [];
  const v = new THREE.Vector3();
  for (const s of app.scans.values()) {
    if (members && !members.includes(s.data.id)) continue;
    if (app.hidden.has(s.data.id)) continue;
    if (app.mainOnly && s.data.group !== 0 && s.data.group !== -1) continue;
    if (!s.points) continue;
    const arr = s.points.geometry.attributes.position.array;
    const n = arr.length / 3;
    if (!n) continue;
    const step = Math.max(1, Math.floor(n / 4000));
    const M = scanDisplayMatrix(s);
    for (let i = 0; i < n; i += step) {
      v.fromArray(arr, i * 3).applyMatrix4(M);
      xs.push(v.x);
      ys.push(v.y);
      zs.push(v.z);
    }
  }
  if (!xs.length) return worldBBox(members);
  const q = (a, p) => a[Math.min(a.length - 1, Math.max(0, Math.floor((a.length - 1) * p)))];
  xs.sort((a, b) => a - b);
  ys.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  return new THREE.Box3(
    new THREE.Vector3(q(xs, 0.005), q(ys, 0.005), q(zs, 0.002)),
    new THREE.Vector3(q(xs, 0.995), q(ys, 0.995), q(zs, 0.998))
  );
}

function updateOffsets() {
  const list = [...app.scans.values()].filter((s) => s.data.loaded && s.data.bbox);
  list.sort((a, b) => a.data.id.localeCompare(b.data.id, undefined, { numeric: true }));
  if (app.layout === 'grid') {
    const n = list.length;
    const cols = Math.max(1, Math.ceil(Math.sqrt(n)));
    let maxW = 1, maxD = 1;
    for (const s of list) {
      maxW = Math.max(maxW, s.data.bbox[1][0] - s.data.bbox[0][0]);
      maxD = Math.max(maxD, s.data.bbox[1][1] - s.data.bbox[0][1]);
    }
    const gap = 1 + app.spread * 8;
    const cellW = maxW + gap;
    const cellD = maxD + gap;
    list.forEach((s, i) => {
      const r = Math.floor(i / cols);
      const c = i % cols;
      const bb = s.data.bbox;
      s.offset = new THREE.Matrix4().makeTranslation(
        c * cellW - bb[0][0],
        r * cellD - bb[0][1],
        -bb[0][2]
      );
    });
  } else {
    for (const s of list) {
      const off = mat4FromRow(s.data.display_offset || [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
      ]);
      if (app.spread > 0) {
        const idx = list.indexOf(s);
        off.multiply(new THREE.Matrix4().makeTranslation(idx * app.spread * 2, 0, 0));
      }
      s.offset = off;
    }
  }
  for (const s of list) {
    if (!s.points) continue;
    const M = scanDisplayMatrix(s);
    s.points.matrix.copy(M);
    s.points.matrixAutoUpdate = false;
    s.points.matrixWorldNeedsUpdate = true;
    if (s.featureEdge) {
      s.featureEdge.matrix.copy(M);
      s.featureEdge.matrixAutoUpdate = false;
      s.featureEdge.matrixWorldNeedsUpdate = true;
    }
    if (s.featurePlane) {
      s.featurePlane.matrix.copy(M);
      s.featurePlane.matrixAutoUpdate = false;
      s.featurePlane.matrixWorldNeedsUpdate = true;
    }
  }
  rebuildGrid();
  if (app.assist) updateAssistGizmo();
  requestRender();
}

/* Scanner links: a line from one scanner position to the other for every link,
 * and a dot where each scanner stood.  Rebuilt on every render, so it follows
 * layout changes, hidden scans and a scan being dragged. */
const scannerLinks = new THREE.Group();
scannerLinks.renderOrder = 10;
root.add(scannerLinks);

function scannerPosition(id) {
  const s = app.scans.get(id);
  if (!s?.points || !s.offset || app.hidden.has(id)) return null;
  if (app.mainOnly && s.data.group !== 0 && s.data.group !== -1) return null;
  const [x, y, z] = s.data.origin || [0, 0, 0];
  return new THREE.Vector3(x, y, z).applyMatrix4(scanDisplayMatrix(s));
}

// WebGL lines are always 1 px, too thin to see over a point cloud: each link is
// a quad of constant width on screen.
const scannerLineMaterial = new THREE.ShaderMaterial({
  uniforms: { uViewport: { value: new THREE.Vector2(1, 1) }, uWidth: { value: 2.5 } },
  vertexShader: `
    attribute vec3 other;
    attribute vec3 aColor;
    attribute float side;
    uniform vec2 uViewport;
    uniform float uWidth;
    varying vec3 vColor;
    void main() {
      const float NEAR = 0.001;
      vec4 a = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      vec4 b = projectionMatrix * modelViewMatrix * vec4(other, 1.0);
      vColor = aColor;
      if (a.w < NEAR && b.w < NEAR) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
      if (a.w < NEAR) a = mix(a, b, (NEAR - a.w) / (b.w - a.w));
      else if (b.w < NEAR) b = mix(b, a, (NEAR - b.w) / (a.w - b.w));
      vec2 d = (b.xy / b.w - a.xy / a.w) * uViewport;
      float len = length(d);
      vec2 n = len > 1e-5 ? vec2(-d.y, d.x) / len : vec2(0.0, 1.0);
      a.xy += n * side * uWidth / uViewport * a.w;
      gl_Position = a;
    }`,
  fragmentShader: `
    varying vec3 vColor;
    void main() { gl_FragColor = vec4(vColor, 1.0); }`,
  transparent: true,
  depthTest: false,
  depthWrite: false,
  side: THREE.DoubleSide,
});

function updateScannerLinks() {
  for (const child of scannerLinks.children) {
    child.geometry.dispose();
    if (child.material !== scannerLineMaterial) child.material.dispose();
  }
  scannerLinks.clear();
  if (!app.showScannerLinks || !app.state) return;
  const pos = [], other = [], side = [], lineColor = [], index = [], dot = [], dotColor = [];
  const colour = new THREE.Color();
  const white = new THREE.Color('#ffffff');
  // lightened, because the scans are drawn in the plain group colours
  const groupTint = (gid) => colour.set(groupColor(Math.max(gid, 0))).lerp(white, 0.45);
  const isSelected = (p) => app.highlightPair === `${p.a}|${p.b}`;
  // the selected line goes last, so it is drawn on top of the others
  for (const p of [...app.state.pairs].sort((x, y) => isSelected(x) - isSelected(y))) {
    if (!p.accepted) continue;
    const a = scannerPosition(p.a), b = scannerPosition(p.b);
    if (!a || !b) continue;
    if (isSelected(p)) colour.set('#ffffff');
    else if (p.manual) colour.set('#ffb86b');
    else groupTint(app.scans.get(p.a).data.group);
    const n = pos.length / 3;
    for (const [from, to, s] of [[a, b, 1], [a, b, -1], [b, a, -1], [b, a, 1]]) {
      pos.push(from.x, from.y, from.z);
      other.push(to.x, to.y, to.z);
      side.push(s);
      lineColor.push(colour.r, colour.g, colour.b);
    }
    index.push(n, n + 1, n + 2, n, n + 2, n + 3);
  }
  for (const s of app.scans.values()) {
    const at = scannerPosition(s.data.id);
    if (!at) continue;
    if (s.data.group >= 0) groupTint(s.data.group);
    else colour.set('#8a93a6');
    dot.push(at.x, at.y, at.z);
    dotColor.push(colour.r, colour.g, colour.b);
  }
  const lineGeometry = new THREE.BufferGeometry()
    .setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    .setAttribute('other', new THREE.Float32BufferAttribute(other, 3))
    .setAttribute('side', new THREE.Float32BufferAttribute(side, 1))
    .setAttribute('aColor', new THREE.Float32BufferAttribute(lineColor, 3))
    .setIndex(index);
  const dotGeometry = new THREE.BufferGeometry()
    .setAttribute('position', new THREE.Float32BufferAttribute(dot, 3))
    .setAttribute('color', new THREE.Float32BufferAttribute(dotColor, 3));
  const lines = new THREE.Mesh(lineGeometry, scannerLineMaterial);
  const dots = new THREE.Points(dotGeometry,
    new THREE.PointsMaterial({ vertexColors: true, size: 6, sizeAttenuation: false, transparent: true, depthTest: false }));
  lines.renderOrder = 10;
  dots.renderOrder = 11;
  lines.frustumCulled = dots.frustumCulled = false;
  scannerLinks.add(lines, dots);
}

let gridKey = '';
function rebuildGrid() {
  const bb = worldBBox();
  if (!bb) { gridGroup.clear(); gridKey = ''; return; }
  const size = Math.max(bb.max.x - bb.min.x, bb.max.y - bb.min.y, 1);
  const key = `${bb.min.x.toFixed(1)},${bb.min.y.toFixed(1)},${size.toFixed(1)}`;
  if (key === gridKey) return;
  gridKey = key;
  gridGroup.clear();
  const step0 = Math.pow(10, Math.floor(Math.log10(size / 8)));
  const step = step0 >= 1 ? step0 : 1;
  const pad = size * 1.2;
  const cx = (bb.min.x + bb.max.x) / 2;
  const cy = (bb.min.y + bb.max.y) / 2;
  const z = bb.min.z;
  const pts = [];
  const half = Math.ceil(pad / 2 / step) * step;
  for (let x = -half; x <= half + 1e-6; x += step) {
    pts.push(cx + x, cy - half, z, cx + x, cy + half, z);
  }
  for (let y = -half; y <= half + 1e-6; y += step) {
    pts.push(cx - half, cy + y, z, cx + half, cy + y, z);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3));
  const grid = new THREE.LineSegments(
    g,
    new THREE.LineBasicMaterial({ color: 0x1d2534, transparent: true, opacity: 0.85 })
  );
  gridGroup.add(grid);

  const axisLen = Math.max(size * 0.06, step);
  const axes = new THREE.BufferGeometry();
  axes.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(
      [cx, cy, z, cx + axisLen, cy, z, cx, cy, z, cx, cy + axisLen, z, cx, cy, z, cx, cy, z + axisLen],
      3
    )
  );
  axes.setAttribute('color', new THREE.Float32BufferAttribute([
    0.9, 0.3, 0.3, 0.9, 0.3, 0.3,
    0.3, 0.85, 0.4, 0.3, 0.85, 0.4,
    0.35, 0.55, 1.0, 0.35, 0.55, 1.0,
  ], 3));
  const axisLines = new THREE.LineSegments(
    axes,
    new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })
  );
  gridGroup.add(axisLines);
}

function fitAll(members = null) {
  const bb = robustBBox(members);
  if (!bb) {
    views.top.center.set(0, 0, 0);
    views.top.half = 10;
    views.side.center.set(0, 0, 0);
    views.side.half = 10;
    orbit.target.set(0, 0, 0);
    orbit.radius = 30;
    requestRender();
    return;
  }
  const c = bb.getCenter(new THREE.Vector3());
  const s = bb.getSize(new THREE.Vector3());
  const base = Math.max(s.x, s.y, s.z, 1);
  views.top.center.set(c.x, c.y, c.z);
  views.top.half = Math.max(s.x, s.y) * 0.58;
  views.top.distance = base * 4;
  views.side.center.set(c.x, c.y, c.z);
  views.side.half = Math.max(s.x, s.z) * 0.58;
  views.side.distance = base * 4;
  orbit.target.copy(c);
  orbit.radius = base * 1.7;
  app.fitPending = false;
  requestRender();
}

/* --------------------------------------------------------------- mesh / fly */

async function loadMesh(path) {
  if (!path) return;
  try {
    toast('Loading mesh…', '');
    const url = `/api/mesh.bin?path=${encodeURIComponent(path)}`;
    const buf = await (await fetch(url)).arrayBuffer();
    const dv = new DataView(buf);
    const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
    if (magic !== 'E57M') throw new Error('bad mesh payload');
    const nv = dv.getUint32(8, true);
    const nf = dv.getUint32(12, true);
    let off = 16;
    const pos = new Float32Array(buf, off, nv * 3);
    off += nv * 12;
    const col8 = new Uint8Array(buf, off, nv * 3);
    off += nv * 3;
    const idx = new Uint32Array(buf.slice(off, off + nf * 12));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const col = new Float32Array(nv * 3);
    for (let i = 0; i < nv * 3; i++) col[i] = col8[i] / 255;
    geometry.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geometry.setIndex(new THREE.BufferAttribute(idx, 1));
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.86, metalness: 0.02, side: THREE.DoubleSide,
    });
    while (meshGroup.children.length) {
      const old = meshGroup.children.pop();
      old.geometry?.dispose?.();
      old.material?.dispose?.();
    }
    meshGroup.add(new THREE.Mesh(geometry, material));
    app.lastMeshPath = path;
    app.meshLoaded = true;
    meshGroup.visible = true;
    root.visible = false;
    $('#btn-fly').disabled = false;
    requestRender();
    toast(`Mesh loaded (${fmtInt(nv)} vertices). Press Fly or P to switch points/mesh.`, 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
}

function enterFly() {
  if (app.fly) {
    exitFly();
    return;
  }
  const dir = new THREE.Vector3();
  threeCam.getWorldDirection(dir);
  app.flyPos.copy(threeCam.position);
  app.flyYaw = Math.atan2(dir.y, dir.x);
  app.flyPitch = Math.asin(clamp(dir.z, -1, 1));
  app.fly = true;
  app.flyLast = performance.now();
  $('#fly-hint').classList.remove('hidden');
  $('#btn-fly').textContent = 'Exit fly';
  try { canvas.requestPointerLock?.(); } catch {}
  requestRender();
}

function exitFly() {
  if (!app.fly) return;
  app.fly = false;
  app.flyKeys.clear();
  $('#fly-hint').classList.add('hidden');
  $('#btn-fly').textContent = 'Fly through';
  try { document.exitPointerLock?.(); } catch {}
  const dir = new THREE.Vector3(
    Math.cos(app.flyPitch) * Math.cos(app.flyYaw),
    Math.cos(app.flyPitch) * Math.sin(app.flyYaw),
    Math.sin(app.flyPitch)
  );
  orbit.target.copy(app.flyPos).addScaledVector(dir, 12);
  orbit.radius = 12;
  orbit.theta = Math.atan2(orbit.target.y - app.flyPos.y, orbit.target.x - app.flyPos.x);
  orbit.phi = Math.acos(clamp(dir.z, -1, 1));
  requestRender();
}

/* ------------------------------------------------------------------ controls */

let drag = null;

// The scanner link under the pointer, within a few pixels of its line.
function scannerLinkAt(clientX, clientY) {
  const pane = paneAt(clientX, clientY);
  if (!pane || !app.showScannerLinks || !app.state) return null;
  const r = canvas.getBoundingClientRect();
  const x = clientX - r.left, y = clientY - r.top;
  const cam = pane.view === 'three' ? threeCam : views[pane.view].camera;
  const toView = (w) => w.clone().applyMatrix4(root.matrixWorld).applyMatrix4(cam.matrixWorldInverse);
  const toPane = (v) => {
    const n = v.clone().applyMatrix4(cam.projectionMatrix);
    return { x: pane.x + (n.x + 1) / 2 * pane.w, y: pane.y + (1 - n.y) / 2 * pane.h };
  };
  let best = null, bestDist = 8;
  for (const p of app.state.pairs) {
    if (!p.accepted) continue;
    const wa = scannerPosition(p.a), wb = scannerPosition(p.b);
    if (!wa || !wb) continue;
    let a = toView(wa), b = toView(wb);
    // clip to the near plane, or a line passing behind the 3D camera flips over
    const near = -cam.near;
    if (a.z > near && b.z > near) continue;
    if (a.z > near) a = a.lerp(b, (a.z - near) / (a.z - b.z));
    else if (b.z > near) b = b.lerp(a, (b.z - near) / (b.z - a.z));
    const pa = toPane(a), pb = toPane(b);
    const dx = pb.x - pa.x, dy = pb.y - pa.y;
    const t = clamp(((x - pa.x) * dx + (y - pa.y) * dy) / Math.max(dx * dx + dy * dy, 1e-6), 0, 1);
    const dist = Math.hypot(x - (pa.x + t * dx), y - (pa.y + t * dy));
    if (dist < bestDist) { bestDist = dist; best = `${p.a}|${p.b}`; }
  }
  return best;
}

function selectLinkInList(key) {
  app.highlightPair = key;
  app.selected = null;
  if (key && app.pairFilter !== 'all' && app.pairFilter !== 'accepted') app.pairFilter = 'all';
  renderPairs();
  renderScans();
  renderSelection();
  requestRender();
  if (key) {
    $('#pair-list .pair.selected')?.scrollIntoView({ block: 'center' });
    $('#scan-list .scan.link-end')?.scrollIntoView({ block: 'nearest' });
  }
}

function paneAt(clientX, clientY) {
  const r = canvas.getBoundingClientRect();
  const x = clientX - r.left;
  const y = clientY - r.top;
  for (const p of panesCSS) {
    if (x >= p.x && x < p.x + p.w && y >= p.y && y < p.y + p.h) return p;
  }
  return null;
}

canvas.addEventListener('pointerdown', (e) => {
  if (app.fly) return;
  if (assistPointerDown(e)) return;
  const pane = paneAt(e.clientX, e.clientY);
  if (!pane) return;
  if (app.cleanupMode && pane.view !== 'three' && e.button === 0) {
    app.cleanupDraft = { pane: pane.view, x0: e.clientX, y0: e.clientY, x1: e.clientX, y1: e.clientY };
    canvas.setPointerCapture(e.pointerId);
    return;
  }
  canvas.setPointerCapture(e.pointerId);
  drag = {
    pane: pane.view,
    x: e.clientX,
    y: e.clientY,
    button: e.button,
    shift: e.shiftKey,
    moved: false,
  };
});

canvas.addEventListener('pointermove', (e) => {
  if (app.fly) {
    if (document.pointerLockElement === canvas) {
      app.flyYaw -= (e.movementX || 0) * 0.0024;
      app.flyPitch = clamp(app.flyPitch - (e.movementY || 0) * 0.0024, -1.45, 1.45);
      requestRender();
    }
    return;
  }
  if (app.assist && assistPointerMove(e)) return;
  if (app.cleanupDraft) {
    app.cleanupDraft.x1 = e.clientX;
    app.cleanupDraft.y1 = e.clientY;
    requestRender();
    return;
  }
  if (!drag) {
    if (app.showScannerLinks && !app.assist && !app.cleanupMode) {
      canvas.style.cursor = scannerLinkAt(e.clientX, e.clientY) ? 'pointer' : '';
    }
    return;
  }
  const dx = e.clientX - drag.x;
  const dy = e.clientY - drag.y;
  drag.x = e.clientX;
  drag.y = e.clientY;
  if (Math.abs(dx) + Math.abs(dy) > 1) drag.moved = true;
  const pane = panesCSS.find((p) => p.view === drag.pane);
  if (!pane) return;

  if (drag.pane === 'three') {
    if (drag.button === 2 || drag.shift || e.shiftKey) {
      // pan: move target in camera plane
      const scale = (orbit.radius * 2.2) / Math.max(pane.h, 1);
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      threeCam.matrixWorld.extractBasis(right, up, new THREE.Vector3());
      orbit.target.addScaledVector(right, -dx * scale);
      orbit.target.addScaledVector(up, dy * scale);
    } else {
      orbit.theta -= dx * 0.006;
      orbit.phi = clamp(orbit.phi - dy * 0.006, 0.05, Math.PI - 0.05);
    }
  } else {
    const view = views[drag.pane];
    const worldPerPx = (2 * view.half) / Math.max(pane.h, 1);
    view.center.x -= dx * worldPerPx;
    if (drag.pane === 'top') view.center.y += dy * worldPerPx;
    else view.center.z += dy * worldPerPx;
  }
  requestRender();
});

canvas.addEventListener('pointerup', async (e) => {
  if (assistPointerUp()) return;
  if (app.cleanupDraft) {
    await commitCleanupDraft();
    drag = null;
    return;
  }
  const click = drag && !drag.moved && drag.button === 0 && !drag.shift;
  drag = null;
  if (click && app.showScannerLinks && !app.assist && !app.cleanupMode) {
    const key = scannerLinkAt(e.clientX, e.clientY);
    if (key || app.highlightPair) selectLinkInList(key);
  }
});
canvas.addEventListener('pointercancel', () => { drag = null; assistPointerUp(); });
canvas.addEventListener('contextmenu', (e) => e.preventDefault());

canvas.addEventListener('wheel', (e) => {
  const pane = paneAt(e.clientX, e.clientY);
  if (!pane) return;
  e.preventDefault();
  const r = canvas.getBoundingClientRect();
  if (pane.view === 'three') {
    orbit.radius = clamp(orbit.radius * Math.exp(e.deltaY * 0.0012), 0.2, 1e6);
    requestRender();
    return;
  }
  const view = views[pane.view];
  const before = view.half;
  view.half = clamp(view.half * Math.exp(e.deltaY * 0.0012), 0.02, 1e6);
  // zoom towards the cursor
  const nx = (e.clientX - r.left - pane.x) / pane.w - 0.5;
  const ny = (e.clientY - r.top - pane.y) / pane.h - 0.5;
  const aspect = pane.w / pane.h;
  let halfH = view.half;
  if (aspect < 1) halfH = view.half / aspect;
  const worldX = nx * 2 * halfH * aspect;
  const worldY = -ny * 2 * halfH;
  const factor = 1 - view.half / before;
  if (pane.view === 'top') {
    view.center.x += worldX * factor;
    view.center.y += worldY * factor;
  } else {
    view.center.x += worldX * factor;
    view.center.z += worldY * factor;
  }
  requestRender();
}, { passive: false });

canvas.addEventListener('dblclick', (e) => {
  const pane = paneAt(e.clientX, e.clientY);
  if (!pane) return;
  if (app.assist) { fitAll([...app.assist.moving, ...app.assist.targetIds]); return; }
  fitAll();
});

window.addEventListener('keydown', (e) => {
  if (!app.fly && assistKeyDown(e)) return;
  const tag = document.activeElement?.tagName;
  const editing = ['INPUT', 'SELECT', 'TEXTAREA'].includes(tag);
  if (app.fly && !editing) {
    const k = e.key.toLowerCase();
    if (['w', 'a', 's', 'd', 'q', 'e', 'shift'].includes(k)) {
      app.flyKeys.add(k);
      e.preventDefault();
    }
    if (k === 'p') {
      root.visible = !root.visible;
      meshGroup.visible = !root.visible;
      requestRender();
    }
    if (k === 'escape') exitFly();
    if (k === 'f') fitAll();
    return;
  }
  if (e.key === 'Escape') {
    $$('.modal-backdrop:not(.hidden)').forEach((m) => m.classList.add('hidden'));
  }
  if ((e.key === 'f' || e.key === 'F') && !e.metaKey && !e.ctrlKey) {
    fitAll(app.assist ? [...app.assist.moving, ...app.assist.targetIds] : null);
  }
});

window.addEventListener('keyup', (e) => {
  if (app.fly) app.flyKeys.delete(e.key.toLowerCase());
});

/* ------------------------------------------------------------------ previews */

async function fetchPreview(scan) {
  if (scan.points || scan.fetching) return;
  scan.fetching = true;
  try {
    const buf = await (await fetch(`/api/scans/${scan.data.id}/preview.bin`)).arrayBuffer();
    const dv = new DataView(buf);
    const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
    if (magic !== 'E57P') throw new Error('bad preview payload');
    const n = dv.getUint32(8, true);
    const flags = dv.getUint32(12, true);
    if (!n) return;
    const pos = new Float32Array(buf, 16, n * 3);
    let off = 16 + n * 12;
    let colors = null;
    if (flags & 1) {
      colors = new Uint8Array(buf, off, n * 3);
      off += n * 3;
    }
    let inten = null;
    if (flags & 2) {
      inten = new Float32Array(buf.slice(off, off + n * 4));
    }
    const attrColor = new Float32Array(n * 3);
    if (colors) {
      for (let i = 0; i < n * 3; i++) attrColor[i] = colors[i] / 255;
    } else {
      const c = hexRGB(scan.data.color);
      for (let i = 0; i < n; i++) {
        attrColor[i * 3] = c.r;
        attrColor[i * 3 + 1] = c.g;
        attrColor[i * 3 + 2] = c.b;
      }
    }
    let intensity = inten;
    if (!intensity) {
      intensity = new Float32Array(n);
    } else {
      let mx = 0;
      for (let i = 0; i < n; i++) mx = Math.max(mx, intensity[i]);
      if (mx > 0) for (let i = 0; i < n; i++) intensity[i] = intensity[i] / mx;
    }

    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('aColor', new THREE.BufferAttribute(attrColor, 3));
    g.setAttribute('aIntensity', new THREE.BufferAttribute(intensity, 1));
    g.computeBoundingSphere();

    const mat = makeMaterial(scan, scan.data.color);
    const points = new THREE.Points(g, mat);
    points.frustumCulled = true;
    points.matrixAutoUpdate = false;
    root.add(points);

    scan.points = points;
    scan.material = mat;
    scan.geometry = g;
    scan.attrColor = attrColor;
    updateOffsets();
    requestRender();
    updateViewStats();
  } catch (err) {
    console.warn('preview failed', scan.data.id, err);
    scan.fetchError = String(err);
  } finally {
    scan.fetching = false;
  }
}

async function fetchFeatures(scan) {
  if (scan.featureEdge || scan.featurePlane || scan.featuresFetching) return;
  scan.featuresFetching = true;
  try {
    const buf = await (await fetch(`/api/scans/${scan.data.id}/features.bin`)).arrayBuffer();
    const dv = new DataView(buf);
    const magic = String.fromCharCode(...new Uint8Array(buf, 0, 4));
    if (magic !== 'E57F') throw new Error('bad features payload');
    const ne = dv.getUint32(4, true);
    const np = dv.getUint32(8, true);
    const eStart = 12;
    const pStart = 12 + ne * 12;
    const mk = (n, start, material) => {
      if (!n) return null;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(buf.slice(start, start + n * 12)), 3));
      geo.setAttribute('aColor', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      geo.setAttribute('aIntensity', new THREE.BufferAttribute(new Float32Array(n), 1));
      geo.computeBoundingSphere();
      const obj = new THREE.Points(geo, material);
      obj.matrixAutoUpdate = false;
      root.add(obj);
      return obj;
    };
    scan.featureEdge = mk(ne, eStart, featureMaterialEdge);
    scan.featurePlane = mk(np, pStart, featureMaterialPlane);
    updateOffsets();
    applyFeatureVisibility();
    updateViewStats();
  } catch (err) {
    console.warn('features failed', err);
  } finally {
    scan.featuresFetching = false;
  }
}

function applyFeatureVisibility() {
  for (const s of app.scans.values()) {
    if (s.featureEdge) s.featureEdge.visible = app.showFeatures;
    if (s.featurePlane) s.featurePlane.visible = app.showFeatures;
  }
  if (app.showFeatures) {
    for (const s of app.scans.values()) {
      if (s.data.loaded) fetchFeatures(s);
    }
  }
  requestRender();
}

function disposeScan(scan) {
  for (const obj of [scan.points, scan.featureEdge, scan.featurePlane]) {
    if (!obj) continue;
    root.remove(obj);
    obj.geometry?.dispose();
    if (obj.material && obj.material !== featureMaterialEdge && obj.material !== featureMaterialPlane) {
      obj.material.dispose();
    }
  }
  app.scans.delete(scan.data.id);
}

/* ------------------------------------------------------------------ state sync */

function stateHash(st) {
  if (!st) return '';
  const scans = st.scans.map((s) => `${s.id}:${s.loaded ? 1 : 0}:${s.visible ? 1 : 0}:${s.included ? 1 : 0}:${s.group}:${s.transform.flat().map((v) => v.toFixed(4)).join(',')}:${s.display_offset.flat().map((v) => v.toFixed(3)).join(',')}`).join(';');
  const pairs = st.pairs.map((p) => `${p.a}|${p.b}|${p.accepted ? 1 : 0}|${(p.score ?? 0).toFixed(3)}|${(p.residual?.overlap ?? -1).toFixed(2)}|${(p.residual?.rot_deg ?? -1).toFixed(1)}|${p.outlier ? 1 : 0}`).join(';');
  const blocked = (st.blocked_links || []).map((p) => `${p.a}|${p.b}`).join(';');
  return `${scans}::${pairs}::${blocked}::${st.params.voxel}::${st.params.accept_score}::${st.reference}`;
}

async function refreshState(renderDom = true) {
  const st = await getJSON('/api/state');
  app.state = st;
  app.activeGroup = st.groups?.length ? st.groups[0].id : 0;
  if (!app.before.size && st.baseline) {
    for (const [sid, m] of Object.entries(st.baseline)) app.before.set(sid, m);
  }

  // remove vanished scans
  const ids = new Set(st.scans.map((s) => s.id));
  for (const [sid, s] of [...app.scans]) {
    if (!ids.has(sid)) disposeScan(s);
  }
  // add / update
  for (const data of st.scans) {
    let s = app.scans.get(data.id);
    if (!s) {
      s = { data, points: null, material: null, offset: new THREE.Matrix4() };
      app.scans.set(data.id, s);
    }
    s.data = data;
  }

  for (const s of app.scans.values()) {
    if (!s.data.loaded && s.points) {
      // settings changed -> server dropped the registration clouds
      for (const obj of [s.points, s.featureEdge, s.featurePlane]) {
        if (!obj) continue;
        root.remove(obj);
        obj.geometry?.dispose();
        if (obj.material && obj.material !== featureMaterialEdge && obj.material !== featureMaterialPlane) obj.material.dispose();
      }
      s.points = s.material = s.geometry = s.featureEdge = s.featurePlane = null;
    }
  }
  updateOffsets();
  for (const s of app.scans.values()) {
    if (s.data.loaded && s.data.preview_count > 0 && !s.points) fetchPreview(s);
  }
  if (app.showFeatures) applyFeatureVisibility();

  const hash = stateHash(st);
  const changed = hash !== app.lastHash;
  app.lastHash = hash;
  if (renderDom && changed) {
    renderScans();
    renderPairs();
    renderGroups();
    renderSelection();
    renderButtons();
    renderParams();
    updateCutTwoButton();
  }
  if (renderDom) {
    const ph = JSON.stringify((st.projects || []).map((p) => [p.id, p.name, p.scans, p.active, p.busy]));
    if (ph !== app.projectsHash) {
      app.projectsHash = ph;
      renderProjects();
    }
  }
  if (app.fitPending) fitAll();
  updateViewStats();
}

function renderProjects() {
  const st = app.state;
  const sel = $('#project-select');
  if (!st) return;
  const projects = st.projects || [];
  sel.innerHTML = projects
    .map((p) => `<option value="${p.id}" ${p.active ? 'selected' : ''}>${escapeHtml(p.name)} · ${p.scans}</option>`)
    .join('');
  renderProjectLock();
}

// Other projects' jobs carry on in the background, so only the active project's job blocks switching.
function renderProjectLock() {
  const st = app.state;
  const sel = $('#project-select');
  sel.disabled = !!st?.busy;
  sel.title = st?.busy
    ? `Switching is off while this project runs a job (${st.busy_kind || 'job'}). Press Stop to end it.`
    : 'Active project';
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function clearScene() {
  for (const s of [...app.scans.values()]) disposeScan(s);
  app.scans.clear();
  app.selected = null;
  app.selection.clear();
  app.transformUndo = null;
  app.highlightPair = null;
  app.before.clear();
  app.hidden.clear();
  app.layout = 'grid';
  app.layoutUserSet = false;
  app.spread = 0;
  const spread = $('#spread');
  if (spread) { spread.value = 0; spread.disabled = false; }
  const gridToggle = $('#toggle-grid');
  if (gridToggle) gridToggle.checked = true;
  app.lastHash = '';
  app.fitPending = true;
  app.focusedGroup = null;
  gridKey = '';
}

function applySavedLayout(st) {
  const hasAlignment = (st.pairs || []).some((p) => p.accepted);
  if (hasAlignment) {
    app.layout = 'true';
    app.layoutUserSet = true;
    $('#toggle-grid').checked = false;
    app.spread = 0;
    $('#spread').value = 0;
    $('#spread').disabled = true;
  } else {
    app.layout = 'grid';
    app.layoutUserSet = false;
    $('#toggle-grid').checked = true;
    $('#spread').disabled = false;
  }
}

async function switchProject(pid) {
  if (!pid) return;
  if (app.job) { toast('Wait for the running job to finish', 'err'); return; }
  try {
    const st = await postJSON('/api/projects/activate', { id: pid });
    clearScene();
    app.state = st;
    app.lastHash = '';
    applySavedLayout(st);
    await refreshState(true);
    fitAll();
    if (st.scans.some((s) => !s.loaded)) {
      await runJob(postJSON('/api/load', {}), 'Loading scans');
      app.fitPending = true;
      fitAll();
    }
  } catch (err) {
    toast(String(err), 'err');
  }
}

function renderButtons() {
  const st = app.state;
  const loaded = st ? st.scans.filter((s) => s.loaded).length : 0;
  const anyScans = st && st.scans.length > 0;
  $('#btn-load').disabled = !anyScans || (loaded === st.scans.length && loaded > 0 && !app.reloadPending);
  const included = st ? st.scans.filter((s) => s.loaded && s.included !== false).length : 0;
  $('#btn-register').disabled = included < 2 || !!app.job || !!st?.busy;
  $('#manual-only').classList.toggle('hidden', !st || st.auto_links !== false || st.locked);
  $('#locked-pill').classList.toggle('hidden', !st?.locked);
  $('#btn-improve').disabled = included < 2 || !!app.job || !!st?.busy || !(st && st.pairs.length);
  if (st?.busy && !app.job) {
    $('#btn-cancel').classList.remove('hidden');
    followJob(st.busy, st.busy_kind);
  } else if (!app.job) {
    $('#btn-cancel').classList.add('hidden');
  }
  renderProjectLock();
  $('#btn-resolve').disabled = !(st && st.pairs.length);
  $('#btn-refine').disabled = !(st && st.pairs.length);
  $('#refine-voxel').disabled = $('#btn-refine').disabled;
  $('#btn-export').disabled = loaded < 1;
  $('#btn-fly').disabled = !anyScans;
  $('#btn-mesh').disabled = false;
  $('#btn-cleanup').disabled = !anyScans;
  $('#btn-clip').disabled = !anyScans;
  $('#empty-state').classList.toggle('hidden', anyScans);
  $('#scan-empty')?.classList.toggle('hidden', anyScans);

  const accepted = st ? st.pairs.filter((p) => p.accepted).length : 0;
  $('#scan-count').textContent = anyScans ? `${loaded}/${st.scans.length}` : '0';
  $('#pair-count').textContent = st ? `${accepted}/${st.pairs.length}` : '0';

  if (st) {
    const groups = st.groups || [];
    const unlinked = groups.filter((g) => g.unconnected).reduce((a, g) => a + g.members.length, 0);
    const parts = [`${loaded} scan${loaded === 1 ? '' : 's'} loaded`];
    if (st.pairs.length) parts.push(`${accepted} links`);
    if (groups.length) parts.push(`${groups.length} group${groups.length === 1 ? '' : 's'}`);
    if (unlinked) parts.push(`${unlinked} unlinked`);
    const totalPts = st.scans.reduce((a, s) => a + (s.preview_count || 0), 0);
    parts.push(`${fmtPts(totalPts)} preview pts`);
    $('#view-stats').textContent = parts.join(' · ');
  }
}

function renderScans() {
  const st = app.state;
  const list = $('#scan-list');
  if (!st || !st.scans.length) {
    list.innerHTML = '<div class="empty" id="scan-empty">No scans yet.<br /><span>Add E57 files or a folder.</span></div>';
    return;
  }
  const groups = st.groups || [];
  const grouped = new Set(groups.flatMap((g) => g.members));
  const unlinked = st.scans.filter((s) => !grouped.has(s.id));

  const linkEnds = app.highlightPair ? app.highlightPair.split('|') : [];
  const row = (s) => {
    const cls = [
      'scan',
      s.id === app.selected ? 'selected' : '',
      linkEnds.includes(s.id) ? 'link-end' : '',
      app.selection.has(s.id) && s.id !== app.selected ? 'multi' : '',
      !s.loaded ? 'unloaded' : '',
    ].join(' ');
    const visible = !app.hidden.has(s.id);
    const included = s.included !== false;
    const g = groups.find((x) => x.members.includes(s.id));
    const color = g ? groupColor(g.id) : '#6b4a4a';
    const badge = !s.loaded ? 'not loaded' : g ? (g.unconnected ? 'unlinked' : `group ${String.fromCharCode(65 + g.id)}`) : 'unlinked';
    return `
      <div class="${cls}" data-sid="${s.id}" style="--gcolor:${color}">
        <input type="checkbox" data-act="include" ${included ? 'checked' : ''} title="include in registration" />
        <button class="eye ${visible ? '' : 'off'}" data-act="eye" title="show / hide">${visible ? '◉' : '◌'}</button>
        <div class="meta">
          <div class="name" title="${escapeHtml(s.name || s.label)}">${escapeHtml(s.label)}</div>
          <div class="sub">
            <span>${fmtPts(s.point_count)} pts</span>
            ${s.cleaned ? `<span title="removed as noise">−${(s.cleaned_pct * 100).toFixed(1)}%</span>` : ''}
            ${s.tilt_deg > 1 ? `<span class="tilt" title="detected gravity ${s.up_mode} · ${Number(s.tilt_deg).toFixed(0)}° from local +Z">tilt ${Number(s.tilt_deg).toFixed(0)}°</span>` : ''}
            ${s.orientation && s.orientation !== 'auto' ? `<span class="tilt" title="orientation override">${s.orientation}</span>` : ''}
            ${s.error ? `<span style="color:#ff9a9a" title="${escapeHtml(s.error)}">error</span>` : ''}
          </div>
        </div>
        ${(!g || g.unconnected) && s.loaded && groups.length ? `<button class="row-place" data-place-sid="${s.id}" data-place-whole="0" title="Drag this scan into place by hand">✋</button>` : ''}
        ${manualPin(s.id)}
        ${s.confidence && s.confidence !== 'unlinked' ? `<span class="conf conf-${s.confidence.replace(' ', '-')}" title="${s.confidence === 'confirmed' ? 'confirmed: its placement is backed by a loop of agreeing links' : 'held by a single link — nothing else confirms it; worth a look'}">${s.confidence === 'confirmed' ? '✓' : '1'}</span>` : ''}
        <span class="badge ${g ? 'ok' : 'warn'}" style="border-color:${color}33;color:${color}">${badge}</span>
      </div>`;
  };

  let html = '';
  for (const g of groups) {
    const members = st.scans.filter((s) => g.members.includes(s.id));
    if (!members.length) continue;
    const c = groupColor(g.id);
    const letter = String.fromCharCode(65 + g.id);
    html += `<div class="scan-group-head" data-gid="${g.id}" style="--gc:${c}"
      title="click to focus · double-click to make reference">
      <span class="gletter">${letter}</span><span>Group ${letter}${g.id === 0 ? ' · main' : ''}</span>
      <span class="gmeta">${members.length} scan${members.length === 1 ? '' : 's'} · ${g.unconnected ? 'unlinked' : `${g.edges} links`}</span>
      ${g.id !== 0 && groups.length > 1 ? `<button class="gh-place" data-place-sid="${g.reference || g.members[0]}" data-place-whole="${g.members.length > 1 ? 1 : 0}" title="Drag this group into place by hand">✋ place</button>` : ''}</div>`;
    html += members.map(row).join('');
  }
  if (unlinked.length) {
    html += `<div class="scan-group-head unlinked" data-gid="unlinked">
      <span class="gletter">?</span><span>Not linked</span><span class="gmeta">${unlinked.length} scans</span></div>`;
    html += unlinked.map(row).join('');
  }
  list.innerHTML = html;
  renderGravity();
  renderCleanupRegions();
  renderCleanupList();
  syncClipControls();
}

function renderGravity() {
  const host = $('#gravity-list');
  if (!host) return;
  const st = app.state;
  if (!st) {
    host.innerHTML = '';
    return;
  }
  const odd = (st.scans || []).filter(
    (s) => s.orientation !== 'auto' || s.up_mode !== 'level' || (s.up_confidence ?? 1) < 0.5
  );
  if (!odd.length) {
    host.innerHTML = '<div class="empty small">All scans look level — gravity is along local +Z.</div>';
    return;
  }
  host.innerHTML =
    odd
      .map((s) => {
        const mode = s.up_mode === 'level' ? 'level' : s.up_mode || 'unknown';
        const conf = Math.round((s.up_confidence ?? 0) * 100);
        return `<div class="gravity-row" data-grav-sid="${s.id}">
          <span class="gname">${escapeHtml(s.label)}</span>
          <span class="gmeta">${mode} · ${conf}% · ${Number(s.tilt_deg || 0).toFixed(0)}°</span>
        </div>`;
      })
      .join('') + '<div class="hint">Click a scan to set its orientation.</div>';
}

function renderCleanupRegions() {
  while (cleanupGroup.children.length) {
    const old = cleanupGroup.children.pop();
    old.geometry?.dispose?.();
    old.material?.dispose?.();
  }
  const regs = app.state?.cleanups || [];
  for (const c of regs) {
    if (c.enabled === false || c.clip) continue;
    let helper;
    if (c.kind === 'sphere') {
      helper = new THREE.Mesh(
        new THREE.SphereGeometry(c.radius || 1, 18, 12),
        new THREE.MeshBasicMaterial({
          color: c.op === 'keep' ? 0x4ecdc4 : 0xff9f45,
          wireframe: true, transparent: true, opacity: 0.55, depthTest: false,
        })
      );
      helper.position.set(c.center[0], c.center[1], c.center[2]);
    } else {
      const size = c.size || [1, 1, 1];
      const box = new THREE.Box3(
        new THREE.Vector3(c.center[0] - size[0], c.center[1] - size[1], c.center[2] - size[2]),
        new THREE.Vector3(c.center[0] + size[0], c.center[1] + size[1], c.center[2] + size[2])
      );
      helper = new THREE.Box3Helper(box, c.op === 'keep' ? 0x4ecdc4 : 0xff9f45);
      helper.material.transparent = true;
      helper.material.opacity = 0.75;
      helper.material.depthTest = false;
    }
    cleanupGroup.add(helper);
  }
  const clip = activeClip();
  const bb = clip ? robustBBox() : null;
  if (clip && bb) {
    // unbounded sides are drawn at the visible scene extent
    const lo = clip.min.map((v, i) => (bounded(v) ? v : bb.min.getComponent(i) - 0.3));
    const hi = clip.max.map((v, i) => (bounded(v) ? v : bb.max.getComponent(i) + 0.3));
    const helper = new THREE.Box3Helper(
      new THREE.Box3(new THREE.Vector3(...lo), new THREE.Vector3(...hi)),
      0x4ecdc4
    );
    helper.material.transparent = true;
    helper.material.opacity = 0.6;
    helper.material.depthTest = false;
    cleanupGroup.add(helper);
  }
  requestRender();
}

function renderCleanupList() {
  const host = $('#cleanup-list');
  if (!host) return;
  const regs = (app.state?.cleanups || []).filter((c) => !c.clip);
  if (!regs.length) {
    host.innerHTML = '<div class="empty small">No regions. Press Remove points, then drag a box.</div>';
    return;
  }
  const active = regs.filter((c) => c.enabled !== false).length;
  const note = regs.length > 32 ? `<div class="hint">preview shows the first 32 regions; export applies all ${regs.length}.</div>` : '';
  host.innerHTML = `<div class="hint" style="margin:0 0 3px">${active} active region${active === 1 ? '' : 's'} · points inside are hidden in the preview and export</div>` + regs
    .map((c) => {
      const what = c.kind === 'sphere' ? `sphere r${Number(c.radius || 1).toFixed(1)}` : 'box';
      const scope = c.scans && c.scans.length ? `${c.scans.length} scan(s)` : 'all scans';
      return `<div class="cleanup-row" data-cid="${c.id}">
        <input type="checkbox" ${c.enabled === false ? '' : 'checked'} data-cleanup-enable />
        <span>${c.op === 'keep' ? 'keep' : 'remove'} · ${what}</span>
        <span class="cmeta">${scope}</span>
        <button class="x" data-cleanup-delete title="delete region">×</button>
      </div>`;
    })
    .join('');
}

function worldFromOrtho(paneName, clientX, clientY) {
  const pane = panesCSS.find((p) => p.view === paneName);
  const view = views[paneName];
  if (!pane || !view) return null;
  const r = canvas.getBoundingClientRect();
  const x = clientX - r.left;
  const y = clientY - r.top;
  const aspect = pane.w / pane.h;
  let halfH = view.half;
  if (aspect < 1) halfH = view.half / aspect;
  const halfW = halfH * aspect;
  const u = (x - pane.x) / pane.w - 0.5;
  const v = (y - pane.y) / pane.h - 0.5;
  return {
    x: view.center.x + u * 2 * halfW,
    y: view.center.y - v * 2 * halfH,
    z: view.center.z - v * 2 * halfH,
  };
}

async function commitCleanupDraft() {
  const d = app.cleanupDraft;
  app.cleanupDraft = null;
  if (!d) return;
  const a = worldFromOrtho(d.pane, d.x0, d.y0);
  const b = worldFromOrtho(d.pane, d.x1, d.y1);
  if (!a || !b) return;
  const bb = robustBBox();
  if (!bb) return;
  const x0 = Math.min(a.x, b.x), x1 = Math.max(a.x, b.x);
  const y0 = Math.min(a.y, b.y), y1 = Math.max(a.y, b.y);
  const z0 = Math.min(a.z, b.z), z1 = Math.max(a.z, b.z);
  let center, size;
  if (d.pane === 'top') {
    center = [(x0 + x1) / 2, (y0 + y1) / 2, (bb.min.z + bb.max.z) / 2];
    size = [Math.max((x1 - x0) / 2, 0.1), Math.max((y1 - y0) / 2, 0.1), Math.max((bb.max.z - bb.min.z) / 2, 0.1)];
  } else {
    center = [(x0 + x1) / 2, (bb.min.y + bb.max.y) / 2, (z0 + z1) / 2];
    size = [Math.max((x1 - x0) / 2, 0.1), Math.max((bb.max.y - bb.min.y) / 2, 0.1), Math.max((z1 - z0) / 2, 0.1)];
  }
  if (Math.max(size[0], size[1]) < 0.15) {
    toast('Drag a larger box', 'err');
    return;
  }
  const scans = app.selection.size ? [...app.selection] : null;
  try {
    await postJSON('/api/cleanup/add', {
      kind: 'box', center, size, op: 'delete', enabled: true, scans,
      label: `${d.pane} box`,
    });
    await refreshState(true);
    toast('Region added: points are hidden in the preview and left out of exports', 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
}

function updateCutTwoButton() {
  const btn = document.querySelector('#btn-cut-two');
  if (!btn) return;
  const ids = [...app.selection];
  btn.classList.toggle('hidden', ids.length < 2);
  if (ids.length >= 2) {
    const labels = ids.map((id) => app.state?.scans.find((s) => s.id === id)?.label || id);
    btn.textContent = `cut ${ids.length}: ${labels.slice(0, 3).join(' + ')}${labels.length > 3 ? ' +…' : ''}`;
  }
}

function renderPairs() {
  const st = app.state;
  const list = $('#pair-list');
  const blockedLinks = st?.blocked_links || [];
  if (st) {
    const all = st.pairs.length + blockedLinks.length;
    const linked = st.pairs.filter((p) => p.accepted).length;
    const sel = $('#pair-filter');
    if (sel) {
      sel.innerHTML = `
        <option value="all">all ${all}</option>
        <option value="accepted">linked ${linked}</option>
        <option value="rejected">rejected by you ${st.pairs.length - linked}</option>
        <option value="blocked">blocked ${blockedLinks.length}</option>`;
      sel.value = app.pairFilter;
    }
  }
  if (!st || (!st.pairs.length && !blockedLinks.length)) {
    list.innerHTML = '<div class="empty" id="pair-empty">No links yet.<br /><span>Press Register to find them.</span></div>';
    return;
  }
  let pairs = st.pairs;
  if (app.pairFilter === 'accepted') pairs = pairs.filter((p) => p.accepted);
  if (app.pairFilter === 'rejected') pairs = pairs.filter((p) => !p.accepted);
  if (app.pairFilter === 'blocked') pairs = [];
  const label = (id) => st.scans.find((s) => s.id === id)?.label || id;
  list.innerHTML = pairs
    .map((p) => {
      const key = `${p.a}|${p.b}`;
      const score = p.score ?? p.overlap ?? 0;
      const sel = app.highlightPair === key ? 'selected' : '';
      const outlier = p.outlier ? 'outlier' : '';
      const residual = p.residual
        ? ` · res ${p.residual.rot_deg != null ? p.residual.rot_deg.toFixed(1) + '°' : ''}${p.residual.trans != null ? ' ' + p.residual.trans.toFixed(2) + 'm' : ''}${p.residual.overlap != null ? ' ov ' + (p.residual.overlap * 100).toFixed(0) + '%' : ''}`
        : '';
      const fm = p.feature_matches;
      const feat = fm && fm.edges_total
        ? ` · ${fm.edges_matched}/${fm.edges_total} edges`
        : '';
      const mm = p.accuracy && p.accuracy.mean_mm != null ? ` · ${Number(p.accuracy.mean_mm).toFixed(1)} mm` : '';
      const wall = p.wall_score != null ? ` · wall ${(Number(p.wall_score) * 100).toFixed(0)}%${p.flat_only ? ' flat' : ''}` : '';
      const corner = p.corner_matches ? ` · ${p.corner_matches} corners` : '';
      const verify = p.ambiguous ? ' · ambiguous' : (p.rejection_reason && !p.accepted ? ' · weak' : '');
      const cyc = p.cycle && ((p.cycle.min_trans ?? 0) > 0.5 || (p.cycle.min_rot ?? 0) > 8)
        ? ` · <span style="color:#ffb86b" title="best alternate two-hop path disagrees with this link">cycle ${Number(p.cycle.min_trans ?? 0).toFixed(2)}m/${Number(p.cycle.min_rot ?? 0).toFixed(1)}°</span>`
        : '';
      const why = p.auto_rejected && p.rejected_reason ? ` · <span class="why">${escapeHtml(p.rejected_reason)}</span>` : '';
      return `
        <div class="pair ${p.accepted ? 'linked' : 'unlinked'} ${outlier} ${sel}" data-key="${key}">
          <div class="names" title="${label(p.a)} ↔ ${label(p.b)}">${label(p.a)} <b>↔</b> ${label(p.b)}</div>
          <div class="score">${(score * 100).toFixed(0)}%</div>
          <div class="bar"><i style="width:${(score * 100).toFixed(1)}%"></i></div>
          <div class="details" title="${escapeHtml(p.method || '')}">${methodLabel(p.method)}${feat}${wall}${corner}${verify}${mm}${residual}${cyc}${p.outlier ? ' · <span style="color:#ff9a6a">outlier ' + (p.outlier_reason || '') + '</span>' : ''}${why}</div>
          <div class="pair-actions">
            <button class="link-toggle ${p.accepted ? 'on' : ''}" data-act="accept" title="${p.accepted ? 'click to reject this link' : 'click to accept this link'}">${p.accepted ? 'linked' : 'unlinked'}</button>
            <button class="mini" data-act="inspect" title="open the cross-section view to check this link">check</button>
            <button class="mini danger" data-act="block" title="Never link these two scans again, in any project that has both. Use this for a false match: it stays out of Register, Improve links and Re-choose links until you unblock it.">block</button>
          </div>
        </div>`;
    })
    .join('') + (app.pairFilter === 'all' || app.pairFilter === 'blocked' ? blockedLinks : [])
    .map((p) => `
        <div class="pair blocked" data-key="${p.a}|${p.b}">
          <div class="names" title="${label(p.a)} ↔ ${label(p.b)}">${label(p.a)} <b>↔</b> ${label(p.b)}</div>
          <div class="details">Blocked: these two scans are never linked</div>
          <div class="pair-actions">
            <button class="mini" data-act="unblock" title="Allow these two scans to be linked again, then choose links again">unblock</button>
          </div>
        </div>`)
    .join('');
}

// Plain name for how a link was found; the exact string stays in the tooltip.
function methodLabel(method) {
  const m = String(method || '');
  if (!m) return '';
  if (m.startsWith('manual')) return 'Placed by hand';
  if (m.startsWith('model snap')) return 'Fitted to the joined scans';
  if (m.startsWith('local') && m.includes('floorplan')) return 'Floor-plan search near the scanners';
  if (m.includes('floorplan')) return 'Floor-plan search';
  return m;
}

const GROUP_COLORS = [
  '#4ecdc4', '#ffd93d', '#a78bfa', '#ff9f45', '#6bcb77', '#4d96ff',
  '#ff6b6b', '#f78fb3', '#00d2d3', '#eccc68', '#c56cf0', '#7bed9f',
];

function groupColor(gid) {
  return GROUP_COLORS[Number(gid) % GROUP_COLORS.length];
}

function renderGroups() {
  const st = app.state;
  const host = $('#group-chips');
  if (!st || !host) return;
  const groups = st.groups || [];
  const all = st.scans || [];
  const grouped = new Set(groups.flatMap((g) => g.members));
  const unlinked = all.filter((s) => !grouped.has(s.id));
  const activeAll = app.focusedGroup === null || app.focusedGroup === undefined;
  let html = `<button class="group-chip ${activeAll ? 'active' : ''}" data-gid="all" style="--gc:var(--accent)">
    <b>All</b><span class="g-meta">${all.length} scans</span></button>`;
  for (const g of groups) {
    const c = groupColor(g.id);
    const letter = String.fromCharCode(65 + g.id);
    html += `<button class="group-chip ${app.focusedGroup === g.id ? 'active' : ''}" data-gid="${g.id}" style="--gc:${c}"
      title="click to focus · double-click to make reference">
      <span class="g-dot"></span><b>${letter}</b> ${g.members.length}
      <span class="g-meta">${g.unconnected ? 'unlinked' : `${g.edges} links`}${g.id === 0 ? ' · main' : ''}</span></button>`;
  }
  if (unlinked.length) {
    html += `<button class="group-chip unlinked ${app.focusedGroup === 'unlinked' ? 'active' : ''}" data-gid="unlinked">
      <b>Unlinked</b><span class="g-meta">${unlinked.length}</span></button>`;
  }
  host.innerHTML = html;
  const accepted = (st.pairs || []).filter((p) => p.accepted).length;
  const acc = st.accuracy;
  const accTxt = acc && acc.mean_mm != null ? ` · ${Number(acc.mean_mm).toFixed(1)} mm mean` : '';
  $('#strip-summary').textContent =
    `${all.length} scans · ${groups.length} group${groups.length === 1 ? '' : 's'} · ${accepted} links${accTxt}`;
}

function focusGroup(gid) {
  const st = app.state;
  if (!st) return;
  if (gid === 'all' || gid === null) {
    app.focusedGroup = null;
    app.hidden.clear();
    for (const s of app.scans.values()) {
      if (s.points) s.points.visible = true;
      if (s.featureEdge) s.featureEdge.visible = app.showFeatures;
      if (s.featurePlane) s.featurePlane.visible = app.showFeatures;
    }
    renderGroups();
    renderScans();
    fitAll();
    requestRender();
    return;
  }
  let ids = [];
  if (gid === 'unlinked') {
    const grouped = new Set((st.groups || []).flatMap((g) => g.members));
    ids = st.scans.filter((s) => !grouped.has(s.id)).map((s) => s.id);
    app.focusedGroup = 'unlinked';
  } else {
    const g = (st.groups || []).find((x) => String(x.id) === String(gid));
    if (!g) return;
    ids = g.members;
    app.focusedGroup = g.id;
  }
  app.hidden = new Set(st.scans.filter((s) => !ids.includes(s.id)).map((s) => s.id));
  for (const [id, obj] of app.scans) {
    const vis = !app.hidden.has(id);
    if (obj?.points) obj.points.visible = vis;
    if (obj?.featureEdge) obj.featureEdge.visible = vis && app.showFeatures;
    if (obj?.featurePlane) obj.featurePlane.visible = vis && app.showFeatures;
  }
  renderGroups();
  renderScans();
  fitAll(ids);
  requestRender();
}

async function setReferenceForGroup(gid) {
  const g = (app.state?.groups || []).find((x) => String(x.id) === String(gid));
  if (!g) return;
  await runJob(postJSON('/api/reference', { id: g.reference }), 'Setting reference');
  app.fitPending = true;
  fitAll();
}

function renderSelection() {
  const host = $('#selection');
  const st = app.state;
  let pairHtml = '';
  if (app.highlightPair && st) {
    const [pa, pb] = app.highlightPair.split('|');
    const pair = st.pairs.find((p) => p.a === pa && p.b === pb);
    if (pair) {
      const la = st.scans.find((s) => s.id === pa)?.label || pa;
      const lb = st.scans.find((s) => s.id === pb)?.label || pb;
      const fm = pair.feature_matches || {};
      const res = pair.residual || {};
      pairHtml = `
        <div class="sel-title">Link ${la} ↔ ${lb}</div>
        <div class="kv">
          <span>score</span><span>${((pair.score ?? 0) * 100).toFixed(0)}%</span>
          <span>structure</span><span>${((pair.structure_overlap ?? 0) * 100).toFixed(0)}%</span>
          <span>raw overlap</span><span>${((pair.overlap ?? 0) * 100).toFixed(0)}%</span>
          <span>fit error</span><span>${(pair.rmse ?? 0).toFixed(3)} m</span>
          <span>found by</span><span title="${escapeHtml(pair.method || '')}">${methodLabel(pair.method) || '—'}</span>
          <span>edges</span><span>${fm.edges_matched ?? 0}/${fm.edges_total ?? 0}</span>
          <span>planes</span><span>${fm.planes_matched ?? 0}/${fm.planes_total ?? 0}</span>
          <span>residual</span><span>${res.rot_deg != null ? res.rot_deg.toFixed(1) + '°' : '—'} ${res.trans != null ? res.trans.toFixed(2) + ' m' : ''}</span>
          <span>surface</span><span>${pair.accuracy && pair.accuracy.mean_mm != null ? `${Number(pair.accuracy.mean_mm).toFixed(1)} mm mean · p95 ${Number(pair.accuracy.p95_mm).toFixed(1)}` : 'not measured'}</span>
          <span>wall bands</span><span>${pair.wall_score != null ? `${(Number(pair.wall_score) * 100).toFixed(0)}%${pair.flat_only ? ' · flat-only / underconstrained' : ''}` : '—'}</span>
          <span>corners</span><span>${pair.corner_matches != null ? `${pair.corner_matches} matched${pair.corner_frac ? ` · ${Math.round(pair.corner_frac * 100)}%` : ''}` : '—'}</span>
          <span>verification</span><span>${pair.accepted ? (pair.ambiguous ? 'ambiguous margin' : 'accepted') : (pair.rejection_reason || 'rejected')}${pair.margin != null ? ` · margin ${Number(pair.margin).toFixed(3)}` : ''}</span>
          <span>loop check</span><span style="color:${(pair.cycle?.min_trans ?? 0) > 0.5 || (pair.cycle?.min_rot ?? 0) > 8 ? '#ffb86b' : 'inherit'}">${
            pair.cycle
              ? `${Number(pair.cycle.min_trans ?? 0).toFixed(2)} m / ${Number(pair.cycle.min_rot ?? 0).toFixed(1)}°${(pair.cycle.min_trans ?? 0) > 0.5 || (pair.cycle.min_rot ?? 0) > 8 ? ' · inconsistent' : ''}`
              : '—'
          }</span>
          <span>alternatives</span><span>${(pair.alternatives || []).length > 1 ? (pair.alternatives || []).map((a) => `${methodLabel(a.method)} ${Math.round(a.score * 100)}%`).join(' · ') : '—'}</span>
        </div>
        <div class="btn-row">
          <button class="btn small" data-pair="toggle">${pair.accepted ? 'Reject link' : 'Accept link'}</button>
          <button class="btn small primary" data-pair="inspect">Check in cross-section…</button>
          <button class="btn small" data-pair="measure">Measure mm</button>
        </div>
        <div class="hint">An accepted link is pinned as a manual link. A rejected position stays out of later runs.</div>`;
      if (!app.selected) {
        host.innerHTML = pairHtml;
        return;
      }
      pairHtml += '<div class="panel-head sub" style="padding-left:0;padding-right:0"><h2>Scan</h2></div>';
    }
  }
  const data = st?.scans.find((s) => s.id === app.selected);
  if (!data) {
    host.innerHTML = pairHtml || '<div class="empty small">Click a scan to inspect it.</div>';
    return;
  }
  const m = mat4FromRow(data.transform);
  const e = new THREE.Euler().setFromRotationMatrix(m, 'XYZ');
  const t = new THREE.Vector3();
  m.decompose(t, new THREE.Quaternion(), new THREE.Vector3());
  const bb = data.bbox;
  const dims = bb ? bb[1].map((v, i) => v - bb[0][i]) : null;
  const isRef = st.reference === data.id;
  const gSel = groupOf(data.id);
  const placeHint = !gSel || gSel.unconnected
    ? 'Not linked yet — drag it into place by hand.'
    : gSel.id !== 0
      ? `Group ${String.fromCharCode(65 + gSel.id)} is separate — place it into the main group.`
      : 'Misplaced? Drag it to the right spot.';
  host.innerHTML = pairHtml + `
    <div class="sel-title">${data.label} ${isRef ? '<span class="badge ok">reference</span>' : ''}</div>
    <div class="sel-sub">${data.file}</div>
    <button class="btn primary wide place-cta" data-sel="place" ${data.loaded ? '' : 'disabled'}>✋ Place manually…</button>
    <div class="hint" style="margin-bottom:6px">${placeHint}</div>
    <div class="kv">
      <span>points</span><span>${fmtInt(data.point_count)}</span>
      <span>loaded</span><span>${data.loaded ? fmtPts(data.preview_count) + ' pts preview' : 'no'}</span>
      <span>noise removed</span><span>${data.cleaned ? `${fmtInt(data.cleaned)} pts (${(data.cleaned_pct * 100).toFixed(1)}%)` : '—'}</span>
      <span>group</span><span>${data.group >= 0 ? String.fromCharCode(65 + data.group) : '—'}</span>
      <span>size</span><span>${dims ? dims.map((v) => v.toFixed(1)).join(' × ') + ' m' : '—'}</span>
      <span>gravity</span><span>${data.up_mode || 'unknown'}${data.tilt_deg > 1 ? ` · tilt ${Number(data.tilt_deg).toFixed(1)}°` : ''}${data.up_confidence != null ? ` · ${Math.round(data.up_confidence * 100)}%` : ''}</span>
    </div>
    <div class="field" style="margin-top:8px">
      <label>Orientation</label>
      <select id="sel-orientation" title="Auto detects gravity; Level assumes the scanner stood on a flat surface; Free disables gravity constraints (full 6-DOF matching)">
        <option value="auto" ${data.orientation === 'auto' || !data.orientation ? 'selected' : ''}>auto · detect gravity</option>
        <option value="level" ${data.orientation === 'level' ? 'selected' : ''}>level · flat surface (local +Z)</option>
        <option value="free" ${data.orientation === 'free' ? 'selected' : ''}>free · 6-DOF, no gravity</option>
      </select>
    </div>
    <div class="hint">Local → world transform (translation m, rotation deg)</div>
    <div class="tf-grid">
      <input type="number" step="0.01" data-tf="x" value="${t.x.toFixed(3)}" title="X" />
      <input type="number" step="0.01" data-tf="y" value="${t.y.toFixed(3)}" title="Y" />
      <input type="number" step="0.01" data-tf="z" value="${t.z.toFixed(3)}" title="Z" />
      <input type="number" step="0.5" data-tf="rx" value="${rotDeg(e.x).toFixed(2)}" title="rotation X" />
      <input type="number" step="0.5" data-tf="ry" value="${rotDeg(e.y).toFixed(2)}" title="rotation Y" />
      <input type="number" step="0.5" data-tf="rz" value="${rotDeg(e.z).toFixed(2)}" title="rotation Z" />
    </div>
    <div class="field" style="margin-top:8px">
      <label>Recover from</label>
      <select id="recover-from">
        <option value="">auto · strongest link</option>
        ${(st.pairs || [])
          .filter((p) => p.accepted && (p.a === data.id || p.b === data.id))
          .map((p) => {
            const other = p.a === data.id ? p.b : p.a;
            const os = st.scans.find((x) => x.id === other);
            return `<option value="${other}">${escapeHtml(os?.label || other)} · ${Math.round((p.score || 0) * 100)}%</option>`;
          })
          .sort()
          .join('')}
      </select>
    </div>
    <div class="btn-row">
      <button class="btn small" data-sel="ref" ${isRef ? 'disabled' : ''}>Set reference</button>
      <button class="btn small" data-sel="reset">Reset pose</button>
      <button class="btn small" data-sel="undo-transform" ${app.transformUndo && app.transformUndo.sid === data.id ? '' : 'disabled'}>Undo pose</button>
      <button class="btn small" data-sel="recover" title="Re-place this scan from a still-aligned neighbour's accepted link">Recover pose</button>
    </div>
    <div class="hint">If two scans were reset together, recover one from a third scan that is still aligned, then the other from that scan.</div>
    <div class="btn-row">
      <button class="btn small" data-sel="iso">Isolate</button>
      <button class="btn small" data-sel="showall">Show all</button>
    </div>`;
}

function renderParams() {
  const p = app.state?.params;
  if (!p) return;
  const c = app.state?.cache;
  if (c && !app.paramsDirty) {
    $('#cache-status').textContent =
      `Noise removed: ${(c.noise_pct * 100).toFixed(1)}% of points`;
  }
  if (app.paramsDirty) return;
  $('#p-accept').value = p.accept_score;
  $('#p-accept-out').textContent = Number(p.accept_score).toFixed(2);
  $('#p-preview').value = String(p.preview_points);
  $('#p-workers').value = p.pair_workers;
  $('#p-workers-out').textContent = p.pair_workers;
  $('#p-pose').checked = !!p.use_pose;
  $('#p-features').checked = !!p.detect_features;
  $('#p-cleanup').value = p.cleanup || 'light';
  $('#reload-warning').classList.toggle('hidden', !app.reloadPending);
}

function updateViewStats() {
  const n = [...app.scans.values()].reduce((a, s) => a + (s.points ? s.points.geometry.attributes.position.count : 0), 0);
  if (n) {
    const el = $('#view-stats');
    if (!el.textContent.includes('preview pts')) el.textContent = `${fmtPts(n)} preview pts`;
  }
}

/* ------------------------------------------------------------------ jobs */

function setProgress(p, msg) {
  $('#progress-fill').style.width = `${clamp(p * 100, 0, 100)}%`;
  if (msg) $('#status-text').textContent = msg;
  if (!p) {
    $('#progress-stage').style.width = '0';
    $('#jb-pct').textContent = '';
    $('#jb-detail').textContent = '';
    app.jobView = null;
  }
}

function fmtDur(sec) {
  if (sec == null || !isFinite(sec) || sec < 0) return '';
  if (sec < 45) return 'under a minute';
  if (sec < 90) return '~1 min';
  if (sec < 3600) return `~${Math.round(sec / 60)} min`;
  const h = Math.floor(sec / 3600);
  return `~${h} h ${Math.round((sec - h * 3600) / 60)} min`;
}

// Rich job status: title (job + stage), total % and a thin stage bar, a
// detail line (items, stage %, time left) and a hover panel with all stages.
function renderJobBar(j, label) {
  const now = Date.now() / 1000;
  const total = clamp(j.progress || 0, 0, 1);
  const stage = j.stage || '';
  const hasItems = j.total > 0;
  const stageFrac = hasItems ? clamp((j.done || 0) / j.total, 0, 1) : null;
  // time left in this stage, from its own rate (falls back to the job's)
  let stageLeft = null;
  if (hasItems && j.done > 0 && j.stage_started) {
    const el = now - j.stage_started;
    stageLeft = el * (j.total - j.done) / j.done;
  }
  let totalLeft = null;
  const elapsed = now - (j.started || now);
  if (!j.open_ended && total > 0.04 && elapsed > 5) totalLeft = elapsed * (1 - total) / total;
  const firstDone = j.open_ended && (j.stages || []).some((s) => /^Improve/.test(s.name));

  $('#status-text').textContent = stage ? `${label} · ${stage}` : label;
  $('#progress-fill').style.width = `${(firstDone ? 1 : total) * 100}%`;
  $('#progress-stage').style.width = stageFrac == null ? '0' : `${stageFrac * 100}%`;
  $('#jb-pct').textContent = firstDone ? '' : `${Math.round(total * 100)}%`;
  const bits = [];
  if (firstDone) bits.push('first result ready ✓ · improving in the background');
  if (hasItems) bits.push(`${j.done}/${j.total}${stageFrac != null ? ` · stage ${Math.round(stageFrac * 100)}%` : ''}`);
  if (stageLeft != null && (firstDone || totalLeft == null || stageLeft < totalLeft * 0.9)) bits.push(`${fmtDur(stageLeft)} left in stage`);
  if (totalLeft != null) bits.push(`${fmtDur(totalLeft)} left`);
  if (!bits.length && j.message) bits.push(j.message);
  $('#jb-detail').textContent = bits.join(' · ');
  app.jobView = { j, label, stageFrac, stageLeft, totalLeft };
  renderJobPop();
}

function renderJobPop() {
  const pop = $('#jb-pop');
  const v = app.jobView;
  if (!v || pop.classList.contains('hidden')) return;
  const { j, label } = v;
  const hist = j.stages || [];
  const plan = j.plan || [];
  const names = [...plan];
  for (const s of hist) {
    const base = s.name.replace(/^Improve \d+: .*/, 'Improve (background)');
    if (!names.includes(base) && !names.includes(s.name)) names.push(s.name);
  }
  const now = Date.now() / 1000;
  const rows = names.map((n) => {
    const runs = hist.filter((s) => s.name === n || (n === 'Improve (background)' && /^Improve/.test(s.name)));
    const cur = runs.find((s) => s.finished == null && (j.stage === s.name));
    const done = runs.length && runs.every((s) => s.finished != null) && !cur;
    const skipped = !runs.length && hist.some((s) => names.indexOf(s.name) > names.indexOf(n));
    const dur = runs.reduce((a, s) => a + ((s.finished || now) - s.started), 0);
    const cls = cur ? 'now' : done ? 'done' : '';
    const icon = cur ? '▶' : done ? '✓' : skipped ? '–' : '○';
    let right = runs.length ? `${Math.round(dur)} s` : skipped ? 'skipped' : '';
    if (cur && v.stageLeft != null) right = `${fmtDur(v.stageLeft)} left`;
    const sub = cur && j.stage !== n ? `<div class="bar" style="background:none;height:auto"><small>${escapeHtml(j.stage)}</small></div>` : '';
    const bar = cur && v.stageFrac != null ? `<div class="bar"><i style="width:${v.stageFrac * 100}%"></i></div>` : '';
    return `<div class="jb-stage ${cls}"><span>${icon}</span><span>${escapeHtml(n)}${cur && j.total ? ` · ${j.done}/${j.total}` : ''}</span><small>${right}</small>${sub}${bar}</div>`;
  }).join('');
  const elapsed = now - (j.started || now);
  pop.innerHTML = `<h4>${escapeHtml(label)} · ${Math.round(elapsed / 60)} min running</h4>${rows || ''}
    <div class="jb-msg">${escapeHtml(j.message || '')}</div>
    ${j.open_ended ? '<div class="jb-msg">Keeps improving until nothing is left to try or you press Stop — progress is kept.</div>' : ''}`;
}

$('#jobbar').addEventListener('mouseenter', () => {
  if (!app.jobView) return;
  $('#jb-pop').classList.remove('hidden');
  renderJobPop();
});
$('#jobbar').addEventListener('mouseleave', () => $('#jb-pop').classList.add('hidden'));

const JOB_LABELS = {
  register: 'Register', improve: 'Improve links', realign: 'Re-choosing links', rematch: 'Re-match pair', load: 'Loading',
  refine: 'Polish', export: 'Export', accuracy: 'Measuring', 'export-mesh': 'Mesh',
};

// Attach the status bar to a job someone else started (API, other tab).
function followJob(jobId, kind) {
  if (app.job === jobId || app.following === jobId) return;
  app.following = jobId;
  const label = JOB_LABELS[kind] || kind || 'Working';
  runJob(Promise.resolve({ job: jobId }), label, { pollState: true })
    .finally(() => { if (app.following === jobId) app.following = null; });
}

async function runJob(resPromise, label, { pollState = false, onProgress = null } = {}) {
  const res = await resPromise;
  if (!res || !res.job) return null;
  const jobId = res.job;
  app.job = jobId;
  $('#btn-cancel').classList.remove('hidden');
  setProgress(0, label);
  let tick = 0;
  while (true) {
    let j;
    try {
      j = await getJSON(`/api/jobs/${jobId}`);
    } catch (err) {
      app.job = null;
      $('#btn-cancel').classList.add('hidden');
      setProgress(0, `${label} · job lost`);
      toast('Job lost (server restarted?). Progress up to the last checkpoint is saved.', 'err');
      try { await refreshState(true); } catch {}
      return null;
    }
    renderJobBar(j, label);
    if (onProgress) { try { onProgress(j); } catch {} }
    if (pollState && tick % 4 === 0) {
      try { await refreshState(true); } catch {}
    }
    if (j.status === 'done') {
      app.job = null;
      $('#btn-cancel').classList.add('hidden');
      setProgress(1, `${label} · done`);
      await refreshState(true);
      setTimeout(() => setProgress(0, 'Ready'), 1600);
      return j.result;
    }
    if (j.status === 'error') {
      app.job = null;
      $('#btn-cancel').classList.add('hidden');
      setProgress(0, `${label} failed`);
      toast(j.error || 'job failed', 'err');
      await refreshState(true);
      return null;
    }
    tick++;
    await sleep(tick < 6 ? 250 : 500);
  }
}

/* ------------------------------------------------------------------ actions */

async function addScans(paths, force = false) {
  try {
    const res = await postJSON('/api/files', { paths, force });
    const skipped = res.skipped || [];
    if (!res.added || !res.added.length) {
      const why = skipped.length ? skipped[0].reason : 'no E57 files found';
      toast(`Nothing added: ${why}`, 'err');
      await refreshState(true);
      return;
    }
    app.layout = 'grid';
    app.layoutUserSet = false;
    await refreshState(true);
    app.fitPending = true;
    fitAll();
    if (skipped.length) {
      toast(`${res.added.length} file(s) added · ${skipped.length} duplicate(s) skipped (${skipped[0].reason})`, '');
    }
    const loadRes = await runJob(postJSON('/api/load', {}), 'Loading scans');
    if (loadRes) {
      app.fitPending = true;
      fitAll();
    }
  } catch (err) {
    toast(String(err), 'err');
  }
}

function showRegistered() {
  if (!app.layoutUserSet) {
    app.layout = 'true';
    $('#toggle-grid').checked = false;
    app.spread = 0;
    $('#spread').value = 0;
    $('#spread').disabled = true;
    updateOffsets();
  }
  app.fitPending = true;
  fitAll();
}

// One button: floor-plan candidates -> global selection -> refine (minutes),
// then (optionally) every heavy matcher where links are missing or unsure.
async function registerAll() {
  const st = app.state;
  if (!st) return;
  if (app.job || st.busy) { toast('A job is already running — press Stop first.', 'err'); return; }
  const ids = st.scans.filter((s) => s.loaded && s.included !== false).map((s) => s.id);
  if (ids.length < 2) { toast('Add at least two scans first', 'err'); return; }
  if (!app.before.size) {
    for (const s of st.scans) if (s.loaded) app.before.set(s.id, s.transform);
  }
  const clean = $('#start-clean').checked;
  if (clean && !confirm(`Start clean?\n\nStored results for these ${ids.length} scans are deleted and everything is searched again. This includes slow results from earlier "keep improving" runs, also in other projects that use the same scans.`)) return;
  // It deletes things, so it must never stay ticked for the next run.
  $('#start-clean').checked = false;
  const keep = $('#keep-improving').checked;
  toast(keep
    ? 'Registering: first result in a few minutes, then it keeps improving difficult links in the background. Press Stop any time — progress is kept.'
    : 'Registering: floor-plan matching, global link selection and refinement.', '');
  let shown = false;
  const res = await runJob(
    postJSON('/api/register', { fresh: clean, keep_improving: keep }),
    keep ? 'Register + improve' : 'Register',
    {
      pollState: true,
      onProgress: (j) => {
        // show the first result as soon as the global selection is done
        if (!shown && (j.progress || 0) >= 0.8) { shown = true; showRegistered(); }
      },
    }
  );
  if (res && !res.error) {
    showRegistered();
    const groups = res.groups || [];
    const conf = (app.state?.scans || []).reduce((m, s) => { m[s.confidence] = (m[s.confidence] || 0) + 1; return m; }, {});
    const acc = res.accuracy;
    const mm = acc && acc.mean_mm != null ? ` · ${Number(acc.mean_mm).toFixed(1)} mm mean` : '';
    toast(
      `${res.cancelled ? 'Stopped' : 'Done'}: ${res.accepted} links · ${groups.length} group${groups.length === 1 ? '' : 's'} · ` +
      `${conf.confirmed || 0} confirmed, ${conf['single link'] || 0} on one link, ${conf.unlinked || 0} unlinked${mm}`,
      res.cancelled ? '' : 'ok'
    );
  }
}

async function improveNow() {
  if (app.job || app.state?.busy) { toast('A job is already running.', 'err'); return; }
  toast('Improving: trying the slower matchers on unlinked scans, single links and unsure pairs.', '');
  const res = await runJob(postJSON('/api/improve', {}), 'Improve links', { pollState: true });
  if (res && !res.error) toast(`Improve ${res.cancelled ? 'stopped' : 'finished'} after ${res.rounds} round(s)`, 'ok');
}

async function reSolve() {
  const res = await runJob(postJSON('/api/realign', {}), 'Re-choosing links');
  if (res) {
    app.fitPending = true;
    fitAll();
    toast('Links chosen again from the stored results', 'ok');
  }
}

async function refine() {
  const voxel = parseFloat($('#refine-voxel').value) || 0.02;
  const res = await runJob(postJSON('/api/refine', { voxel, rounds: 8 }), `Polishing at ${voxel * 100} cm`);
  const r = res?.refine;
  if (!r) return;
  const mm = (x) => `${x.toFixed(1)} mm`;
  if (r.reason === 'cancelled') {
    toast('Polish stopped. Positions are unchanged.');
  } else if (!r.kept) {
    toast(`Polish didn't improve the fit (${mm(r.before.mean_mm)} → ${mm(r.after.mean_mm)}), so the previous positions were kept`);
  } else if (r.before && r.after) {
    toast(`Polished at ${voxel * 100} cm: mean surface error ${mm(r.before.mean_mm)} → ${mm(r.after.mean_mm)}, ` +
          `worst link ${mm(r.before.worst_mm)} → ${mm(r.after.worst_mm)}`, 'ok');
  } else {
    toast('Polish done', 'ok');
  }
}

/* ------------------------------------------------------------------ DOM events */

$('#btn-cut-two').onclick = () => {
  const ids = [...app.selection];
  if (ids.length < 2) return;
  openLinkInspector(ids);
};

$('#project-select').onchange = (e) => switchProject(e.target.value);
$('#btn-project-new').onclick = async () => {
  const name = prompt('New project name:', 'Project ' + ((app.state?.projects?.length || 0) + 1));
  if (!name) return;
  try {
    clearScene();
    await postJSON('/api/projects', { name });
    await refreshState(true);
    app.fitPending = true;
    fitAll();
    toast(`Project “${name}” created — add scans to it`, 'ok');
  } catch (err) { toast(String(err), 'err'); }
};
$('#btn-project-rename').onclick = async () => {
  const p = app.state?.project;
  if (!p) return;
  const name = prompt('Rename project:', p.name);
  if (!name || name === p.name) return;
  try {
    await postJSON('/api/projects/rename', { id: p.id, name });
    await refreshState(true);
  } catch (err) { toast(String(err), 'err'); }
};
$('#btn-project-delete').onclick = async () => {
  const p = app.state?.project;
  if (!p) return;
  if (!confirm(`Delete project “${p.name}” and its alignment? The E57 files are not deleted.`)) return;
  try {
    clearScene();
    await postJSON(`/api/projects/${p.id}`, undefined, 'DELETE');
    await refreshState(true);
    app.fitPending = true;
    fitAll();
    toast('Project deleted', 'ok');
  } catch (err) { toast(String(err), 'err'); }
};

$('#btn-add').onclick = openBrowser;
$('#btn-add-2').onclick = openBrowser;
$('#btn-load').onclick = async () => {
  await runJob(postJSON('/api/load', { force: app.reloadPending }), 'Loading scans');
  app.reloadPending = false;
  $('#reload-warning').classList.add('hidden');
  app.fitPending = true;
  fitAll();
};
$('#btn-register').onclick = registerAll;
$('#locked-pill').onclick = async () => {
  if (!confirm('Unlock this project? Its placement is the benchmark reference; unlocking allows changes again.')) return;
  try { await postJSON('/api/lock', { locked: false }); app.lastHash = ''; await refreshState(true); toast('Project unlocked', 'ok'); }
  catch (err) { toast(errText(err), 'err'); }
};
$('#btn-improve').onclick = improveNow;
$('#btn-fly').onclick = enterFly;
$('#btn-mesh').onclick = () => {
  const path = app.lastMeshPath || prompt('Path to a mesh file (.ply / .obj / .glb):');
  if (path) loadMesh(path.trim());
};
$('#btn-cancel').onclick = async () => {
  const jid = app.job || app.state?.busy;
  if (!jid) return;
  try {
    await postJSON(`/api/jobs/${jid}/cancel`, {});
    setProgress(0, 'stopping…');
    toast('Stopping workers and finishing the current step…', '');
  } catch (err) {
    toast(String(err), 'err');
  }
};
$('#btn-resolve').onclick = reSolve;
$('#btn-refine').onclick = refine;
$('#btn-export').onclick = openExport;
$('#btn-fit').onclick = () => fitAll();
$('#btn-measure').onclick = async () => {
  const res = await runJob(postJSON('/api/measure', {}), 'Measuring surface deviation');
  if (res) {
    await refreshState(true);
    const acc = app.state?.accuracy;
    toast(acc ? `Measured ${acc.links} links · ${Number(acc.mean_mm).toFixed(1)} mm mean` : 'Measured', 'ok');
  }
};
$('#btn-snapshot').onclick = async () => {
  try {
    const res = await postJSON('/api/snapshot', { label: 'manual' });
    toast(`Snapshot saved: ${String(res.path || '').split('/').pop()}`, 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
};
$('#btn-restore').onclick = async () => {
  const path = prompt('Path to a saved session.json file:');
  if (!path) return;
  try {
    setProgress(0.2, 'restoring session');
    await postJSON('/api/session/load', { path: path.trim() });
    app.before.clear();
    app.layout = 'true';
    app.layoutUserSet = true;
    $('#toggle-grid').checked = false;
    app.fitPending = true;
    app.lastHash = '';
    await refreshState(true);
    fitAll();
    setProgress(1, 'session restored');
    setTimeout(() => setProgress(0, 'Ready'), 1500);
    toast('Session restored', 'ok');
  } catch (err) {
    toast(String(err), 'err');
    setProgress(0, 'Ready');
  }
};

$('#btn-backup').onclick = async () => {
  if (!confirm('Restore the previous autosave of this project? Current changes since that save are lost.')) return;
  try {
    setProgress(0.2, 'restoring backup');
    clearScene();
    await postJSON('/api/projects/restore-backup', undefined);
    app.lastHash = '';
    await refreshState(true);
    fitAll();
    const st = app.state;
    if (st && st.scans.some((s) => !s.loaded)) {
      await runJob(postJSON('/api/load', {}), 'Loading scans');
    }
    setProgress(0, 'Ready');
    toast('Restored the previous project autosave', 'ok');
  } catch (err) {
    toast(String(err), 'err');
    setProgress(0, 'Ready');
  }
};


$('#btn-reset-links').onclick = async () => {
  if (!confirm('Forget all links and positions? The scans stay loaded and a snapshot is saved first.')) return;
  try {
    await postJSON('/api/reset-registration', { cleanups: false });
    app.before.clear();
    app.hidden.clear();
    app.selected = null;
    app.highlightPair = null;
    $('#start-clean').checked = true;
    await refreshState(true);
    fitAll();
    toast('Links and positions cleared. "Start clean" is ticked, so Register will search from scratch. Untick it to reuse stored results.', 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
};

$('#btn-clear').onclick = async () => {
  if (!confirm('Remove all scans, links and cleanup regions from this project?')) return;
  await postJSON('/api/clear');
  for (const s of [...app.scans.values()]) disposeScan(s);
  app.before.clear();
  app.hidden.clear();
  app.selected = null;
  app.highlightPair = null;
  app.lastHash = '';
  await refreshState(true);
  fitAll();
};

$('#selection').addEventListener('change', async (e) => {
  if (e.target.id !== 'sel-orientation') return;
  const sid = app.selected;
  if (!sid) return;
  const mode = e.target.value;
  try {
    await postJSON(`/api/scans/${sid}/orientation`, { mode });
    await refreshState(true);
    await runJob(postJSON('/api/load', { ids: [sid] }), 'Rebuilding scan bundle');
    app.fitPending = true;
    fitAll();
    toast(`Orientation set to ${mode}`, 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
});

$('#scan-list').addEventListener('click', async (e) => {
  const place = e.target.closest('[data-place-sid]');
  if (place) {
    e.stopPropagation();
    startAssist(place.dataset.placeSid, { wholeGroup: place.dataset.placeWhole === '1' });
    return;
  }
  const gh = e.target.closest('.scan-group-head');
  if (gh) {
    if (e.detail >= 2) setReferenceForGroup(gh.dataset.gid);
    else focusGroup(gh.dataset.gid);
    return;
  }
  const row = e.target.closest('.scan');
  if (!row) return;
  const sid = row.dataset.sid;
  const act = e.target.dataset.act;
  const s = app.state.scans.find((x) => x.id === sid);
  if (act === 'include') {
    e.stopPropagation();
    const included = e.target.checked;
    s.included = included;
    await postJSON('/api/include', { id: sid, included });
    renderButtons();
    renderScans();
    return;
  }
  if (act === 'eye') {
    e.stopPropagation();
    if (app.hidden.has(sid)) app.hidden.delete(sid);
    else app.hidden.add(sid);
    const obj = app.scans.get(sid);
    const vis = !app.hidden.has(sid);
    if (obj?.points) obj.points.visible = vis;
    if (obj?.featureEdge) obj.featureEdge.visible = vis && app.showFeatures;
    if (obj?.featurePlane) obj.featurePlane.visible = vis && app.showFeatures;
    renderScans();
    requestRender();
    return;
  }
  const additive = e.metaKey || e.ctrlKey;
  if (additive) {
    if (app.selection.has(sid)) app.selection.delete(sid);
    else app.selection.add(sid);
    app.selected = sid;
  } else {
    app.selection = new Set([sid]);
    app.selected = sid;
  }
  app.highlightPair = null;
  renderScans();
  renderSelection();
  updateCutTwoButton();
  updateOffsets();
  applyPointUniforms(true);
});

async function renderSnapshots() {
  const host = $('#snapshots-list');
  if (!host) return;
  try {
    const snaps = await getJSON('/api/snapshots');
    if (!snaps || !snaps.length) {
      host.innerHTML = '<div class="empty small">No snapshots yet.</div>';
      return;
    }
    host.innerHTML = snaps
      .map((r, i) => {
        const when = r.mtime ? new Date(r.mtime * 1000).toLocaleString() : r.name;
        return `<div class="run-row">
          <button class="btn small" data-snap-restore="${i}">restore</button>
          <span>${escapeHtml(r.name)}</span>
          <span class="rmeta">${when}</span>
        </div>`;
      })
      .join('');
    host._snaps = snaps;
  } catch (err) {
    host.innerHTML = `<div class="empty small">${escapeHtml(String(err))}</div>`;
  }
}

$('#snapshots-list').addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-snap-restore]');
  if (!btn) return;
  const snaps = $('#snapshots-list')._snaps || [];
  const snap = snaps[parseInt(btn.dataset.snapRestore, 10)];
  if (!snap) return;
  if (!confirm(`Restore snapshot ${snap.name}?`)) return;
  try {
    await postJSON('/api/session/load', { path: snap.path });
    app.fitPending = true;
    fitAll();
    await refreshState(true);
    toast(`Restored ${snap.name}`, 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
});

async function renderRuns() {
  const host = $('#runs-list');
  if (!host) return;
  try {
    const runs = await getJSON('/api/runs');
    if (!runs || !runs.length) {
      host.innerHTML = '<div class="empty small">No runs yet.</div>';
      return;
    }
    host.innerHTML = runs
      .map((r, i) => {
        const when = r.mtime ? new Date(r.mtime * 1000).toLocaleString() : r.id;
        return `<div class="run-row">
          <button class="btn small" data-run-restore="${i}">restore</button>
          <span>${escapeHtml(r.id)}</span>
          <span class="rmeta">${when}</span>
        </div>`;
      })
      .join('');
    host._runs = runs;
  } catch (err) {
    host.innerHTML = `<div class="empty small">${escapeHtml(String(err))}</div>`;
  }
}

$('#runs-list').addEventListener('click', async (e) => {
  const btn = e.target.closest('[data-run-restore]');
  if (!btn) return;
  const runs = $('#runs-list')._runs || [];
  const run = runs[parseInt(btn.dataset.runRestore, 10)];
  if (!run) return;
  if (!confirm(`Restore ${run.id} into the current project?`)) return;
  try {
    await postJSON('/api/session/load', { path: run.session });
    app.fitPending = true;
    fitAll();
    await refreshState(true);
    toast(`Restored ${run.id}`, 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
});

document.querySelectorAll('details.sect').forEach((d) => {
  d.addEventListener('toggle', () => {
    if (d.open && d.querySelector('#runs-list')) {
      renderSnapshots();
      renderRuns();
    }
  });
});

$('#cleanup-list').addEventListener('click', async (e) => {
  const row = e.target.closest('[data-cid]');
  if (!row) return;
  const cid = row.dataset.cid;
  if (e.target.matches('[data-cleanup-delete]')) {
    await postJSON('/api/cleanup/remove', { id: cid });
    await refreshState(true);
    return;
  }
  if (e.target.matches('[data-cleanup-enable]')) {
    await postJSON('/api/cleanup/update', { id: cid, enabled: e.target.checked });
    await refreshState(true);
  }
});

$('#btn-cleanup').onclick = () => {
  app.cleanupMode = !app.cleanupMode;
  $('#btn-cleanup').classList.toggle('active', app.cleanupMode);
  document.body.classList.toggle('cleanup-mode', app.cleanupMode);
  if (app.cleanupMode) {
    toast('Drag a box in the top or side view. It applies to the selected scans, or all if none are selected.', '');
  }
};

$('#btn-cleanup-clear').onclick = async () => {
  const regs = (app.state?.cleanups || []).filter((c) => !c.clip);
  for (const c of regs) await postJSON('/api/cleanup/remove', { id: c.id });
  await refreshState(true);
};

/* ------------------------------------------------------------------ clip box */

const CLIP_INF = 1e4; // |value| >= this means "unbounded" on that side

function clipRegion() {
  return (app.state?.cleanups || []).find((c) => c.clip) || null;
}

function clipFromRegion(c) {
  if (!c) return null;
  const size = c.size || [CLIP_INF, CLIP_INF, CLIP_INF];
  return {
    enabled: c.enabled !== false,
    min: c.center.map((v, i) => v - size[i]),
    max: c.center.map((v, i) => v + size[i]),
  };
}

/** The clip box to render with ({min, max}) or null when off. */
function activeClip() {
  const c = app.clipDraft || clipFromRegion(clipRegion());
  return c && c.enabled ? c : null;
}

function currentClip() {
  return app.clipDraft || clipFromRegion(clipRegion());
}

function bounded(v) { return Math.abs(v) < CLIP_INF * 0.99; }

/** Sample registration-world points of the visible scans, grouped by overlap group. */
function sampleRegPoints(maxPerScan = 6000) {
  const byGroup = new Map();
  const v = new THREE.Vector3();
  for (const s of app.scans.values()) {
    if (!s.points || app.hidden.has(s.data.id) || !s.data.included) continue;
    if (app.mainOnly && s.data.group !== 0 && s.data.group !== -1) continue;
    const arr = s.points.geometry.attributes.position.array;
    const n = arr.length / 3;
    if (!n) continue;
    const T = mat4FromRow(s.data.transform);
    const step = Math.max(1, Math.floor(n / maxPerScan));
    const key = s.data.group ?? -1;
    if (!byGroup.has(key)) byGroup.set(key, []);
    const out = byGroup.get(key);
    for (let i = 0; i < n; i += step) {
      v.fromArray(arr, i * 3).applyMatrix4(T);
      out.push(v.x, v.y, v.z);
    }
  }
  return byGroup;
}

/**
 * Ceiling height of one point set: the highest strong horizontal-surface peak
 * in the Z histogram (5 cm bins) above the median height.
 */
function ceilingZ(xyz) {
  const zs = [];
  for (let i = 2; i < xyz.length; i += 3) zs.push(xyz[i]);
  if (zs.length < 500) return null;
  zs.sort((a, b) => a - b);
  const lo = zs[Math.floor(zs.length * 0.001)];
  const hi = zs[Math.floor(zs.length * 0.999)];
  const med = zs[Math.floor(zs.length / 2)];
  const bin = 0.05;
  const nb = Math.max(1, Math.ceil((hi - lo) / bin) + 1);
  const h = new Float64Array(nb);
  for (const z of zs) {
    if (z < lo || z > hi) continue;
    h[Math.min(nb - 1, Math.floor((z - lo) / bin))]++;
  }
  const sm = new Float64Array(nb);
  for (let i = 0; i < nb; i++) sm[i] = (h[i - 1] || 0) * 0.25 + h[i] * 0.5 + (h[i + 1] || 0) * 0.25;
  let peak = 0;
  for (let i = 0; i < nb; i++) peak = Math.max(peak, sm[i]);
  for (let i = nb - 1; i >= 0; i--) {
    const z = lo + (i + 0.5) * bin;
    if (z <= med) break;
    const isMax = sm[i] >= (sm[i - 1] || 0) && sm[i] >= (sm[i + 1] || 0);
    if (isMax && sm[i] >= 0.25 * peak) return z;
  }
  return null;
}

function regZRange() {
  let lo = Infinity, hi = -Infinity;
  for (const pts of sampleRegPoints(1500).values()) {
    for (let i = 2; i < pts.length; i += 3) {
      if (pts[i] < lo) lo = pts[i];
      if (pts[i] > hi) hi = pts[i];
    }
  }
  return Number.isFinite(lo) ? [lo, hi] : [-5, 5];
}

function setClipDraft(c, { save = true } = {}) {
  app.clipDraft = c;
  syncClipControls(true);
  renderCleanupRegions();
  if (save) scheduleClipSave();
}

function scheduleClipSave() {
  clearTimeout(app.clipSaveTimer);
  app.clipSaveTimer = setTimeout(saveClip, 350);
}

async function saveClip() {
  const c = app.clipDraft;
  if (!c) return;
  const center = c.min.map((v, i) => (v + c.max[i]) / 2);
  const size = c.min.map((v, i) => Math.max((c.max[i] - v) / 2, 0.01));
  const body = { kind: 'box', op: 'keep', clip: true, center, size, enabled: !!c.enabled, label: 'clip box' };
  try {
    const existing = clipRegion();
    if (existing) await postJSON('/api/cleanup/update', { id: existing.id, ...body });
    else await postJSON('/api/cleanup/add', body);
    if (app.clipDraft === c) app.clipDraft = null; // server state now matches
    await refreshState(true);
  } catch (err) {
    toast(`clip box not saved: ${err}`, 'err');
  }
}

function defaultClip() {
  const [lo, hi] = regZRange();
  return {
    enabled: true,
    min: [-CLIP_INF, -CLIP_INF, lo - 0.5],
    max: [CLIP_INF, CLIP_INF, hi + 0.5],
  };
}

function detectCeiling(base = null) {
  const groups = sampleRegPoints();
  const found = [];
  for (const [gid, pts] of groups) {
    const z = ceilingZ(pts);
    if (z != null) found.push({ gid, z });
  }
  if (!found.length) {
    toast('No ceiling found in the visible scans — drag the Top slider instead', 'err');
    return null;
  }
  // the lowest ceiling drops every ceiling; scans in different groups share
  // a frame only roughly, so report each group's estimate
  const z = Math.min(...found.map((f) => f.z));
  const c = base ? JSON.parse(JSON.stringify(base)) : defaultClip();
  c.enabled = true;
  c.max[2] = +(z - 0.12).toFixed(2);
  if (c.min[2] >= c.max[2]) c.min[2] = c.max[2] - 3;
  const letters = found
    .sort((a, b) => a.gid - b.gid)
    .map((f) => `${f.gid >= 0 ? String.fromCharCode(65 + f.gid) : '?'} ${f.z.toFixed(2)} m`)
    .join(' · ');
  toast(`Ceiling ${letters} → cutting above ${c.max[2].toFixed(2)} m`, 'ok');
  return c;
}

function syncClipControls(force = false) {
  const sect = $('#clip-sect');
  if (!sect) return;
  const editing = sect.contains(document.activeElement) && !force;
  const c = currentClip();
  $('#btn-clip').classList.toggle('active', !!(c && c.enabled));
  const [lo, hi] = regZRange();
  const zmin = Math.floor((Math.min(lo, c ? c.min[2] : lo) - 1) * 10) / 10;
  const zmax = Math.ceil((Math.max(hi, c ? c.max[2] : hi) + 1) * 10) / 10;
  for (const id of ['#clip-top', '#clip-bottom']) {
    $(id).min = zmin;
    $(id).max = zmax;
  }
  const disabled = !c;
  for (const el of sect.querySelectorAll('input, button')) {
    if (el.id !== 'clip-on' && el.id !== 'clip-detect') el.disabled = disabled;
  }
  $('#clip-on').checked = !!(c && c.enabled);
  $('#clip-top-out').textContent = c ? `${c.max[2].toFixed(2)} m` : '—';
  $('#clip-bottom-out').textContent = c ? (bounded(c.min[2]) && c.min[2] > zmin ? `${c.min[2].toFixed(2)} m` : 'none') : '—';
  if (editing || !c) return;
  $('#clip-top').value = c.max[2];
  $('#clip-bottom').value = Math.max(c.min[2], zmin);
  const num = (id, v) => { $(id).value = bounded(v) ? v.toFixed(2) : ''; };
  num('#clip-xmin', c.min[0]);
  num('#clip-xmax', c.max[0]);
  num('#clip-ymin', c.min[1]);
  num('#clip-ymax', c.max[1]);
}

function editClip(fn) {
  const cur = currentClip();
  const c = cur ? JSON.parse(JSON.stringify(cur)) : defaultClip();
  fn(c);
  setClipDraft(c);
}

$('#btn-clip').onclick = () => {
  $('#clip-sect').open = true;
  $('#clip-sect').scrollIntoView({ block: 'nearest' });
  const cur = currentClip();
  if (!cur) {
    const c = detectCeiling();
    setClipDraft(c || defaultClip());
    return;
  }
  editClip((c) => { c.enabled = !c.enabled; });
};

$('#clip-on').onchange = (e) => {
  const on = e.target.checked;
  if (!currentClip() && on) {
    setClipDraft(detectCeiling() || defaultClip());
    return;
  }
  editClip((c) => { c.enabled = on; });
};

$('#clip-top').oninput = (e) => editClip((c) => {
  c.max[2] = Number(e.target.value);
  if (c.min[2] > c.max[2] - 0.05) c.min[2] = c.max[2] - 0.05;
});

$('#clip-bottom').oninput = (e) => editClip((c) => {
  const v = Number(e.target.value);
  // the slider's left end means "no floor cut"
  c.min[2] = v <= Number(e.target.min) + 1e-6 ? -CLIP_INF : Math.min(v, c.max[2] - 0.05);
});

for (const [id, axis, side] of [
  ['#clip-xmin', 0, 'min'], ['#clip-xmax', 0, 'max'],
  ['#clip-ymin', 1, 'min'], ['#clip-ymax', 1, 'max'],
]) {
  $(id).onchange = (e) => editClip((c) => {
    const raw = e.target.value.trim();
    c[side][axis] = raw === '' ? (side === 'min' ? -CLIP_INF : CLIP_INF) : Number(raw);
  });
}

$('#clip-detect').onclick = () => {
  const c = detectCeiling(currentClip());
  if (c) setClipDraft(c);
};

$('#clip-fit-xy').onclick = () => editClip((c) => {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const pts of sampleRegPoints(3000).values()) {
    for (let i = 0; i < pts.length; i += 3) {
      x0 = Math.min(x0, pts[i]); x1 = Math.max(x1, pts[i]);
      y0 = Math.min(y0, pts[i + 1]); y1 = Math.max(y1, pts[i + 1]);
    }
  }
  if (!Number.isFinite(x0)) return;
  c.min[0] = +(x0 - 0.2).toFixed(2); c.max[0] = +(x1 + 0.2).toFixed(2);
  c.min[1] = +(y0 - 0.2).toFixed(2); c.max[1] = +(y1 + 0.2).toFixed(2);
});

$('#clip-clear-xy').onclick = () => editClip((c) => {
  c.min[0] = c.min[1] = -CLIP_INF;
  c.max[0] = c.max[1] = CLIP_INF;
});

$('#clip-remove').onclick = async () => {
  clearTimeout(app.clipSaveTimer);
  app.clipDraft = null;
  const existing = clipRegion();
  if (existing) await postJSON('/api/cleanup/remove', { id: existing.id });
  await refreshState(true);
  renderCleanupRegions();
};

$('#gravity-list').addEventListener('click', (e) => {
  const row = e.target.closest('[data-grav-sid]');
  if (!row) return;
  app.selected = row.dataset.gravSid;
  app.selection = new Set([app.selected]);
  app.highlightPair = null;
  renderScans();
  renderSelection();
  updateCutTwoButton();
  updateOffsets();
  applyPointUniforms(true);
});

$('#pair-list').addEventListener('dblclick', (e) => {
  const row = e.target.closest('.pair');
  if (row) openLinkInspector(row.dataset.key);
});

function pair_accepted(a, b) {
  const p = app.state?.pairs.find((x) => x.a === a && x.b === b);
  return !!(p && p.accepted);
}

async function togglePairLink(a, b, forceValue = null) {
  const next = forceValue === null ? !pair_accepted(a, b) : forceValue;
  const prevFilter = app.pairFilter;
  await postJSON('/api/pairs/accept', { a, b, accepted: next });
  // optimistic local update so the row does not jump or disappear
  const p = app.state?.pairs.find((x) => x.a === a && x.b === b);
  if (p) p.accepted = next;
  app.pairFilter = 'all';
  app.highlightPair = `${a}|${b}`;
  renderPairs();
  renderSelection();
  // re-choose links: a rejected pose stays out, an accepted one is pinned
  await runJob(postJSON('/api/realign', {}), 'Re-choosing links');
  toast(
    next ? 'Link accepted and links chosen again' : 'Link rejected and links chosen again',
    'ok',
    {
      label: 'Undo',
      fn: () => togglePairLink(a, b, !next),
    }
  );
}

async function setLinkBlocked(a, b, blocked) {
  const p = app.state?.pairs.find((x) => x.a === a && x.b === b);
  if (blocked && p?.manual && !confirm('This link was placed by hand. Blocking also withdraws that placement, and unblocking does not bring it back. Block it?')) return;
  try {
    const res = await postJSON('/api/pairs/block', { a, b, blocked });
    if (app.highlightPair === `${a}|${b}`) app.highlightPair = null;
    // no re-choose here: scans stay where they are, so Undo is exact.
    // "Re-choose links" updates the layout when you are ready.
    toast(
      blocked
        ? 'Link blocked. Scans not moved; re-choose links to update the layout'
        : res?.restored
          ? 'Link unblocked and put back as it was'
          : 'Link unblocked. Scans moved or the app restarted since, so re-choose links to link them again',
      'ok',
      { label: 'Undo', fn: () => setLinkBlocked(a, b, !blocked) }
    );
    await refreshState(true);
  } catch (err) {
    toast(String(err), 'err');
  }
}

$('#pair-list').addEventListener('click', async (e) => {
  const row = e.target.closest('.pair');
  if (!row) return;
  const [a, b] = row.dataset.key.split('|');
  const act = e.target.dataset.act;
  if (act === 'block' || act === 'unblock') {
    e.stopPropagation();
    await setLinkBlocked(a, b, act === 'block');
    return;
  }
  if (row.classList.contains('blocked')) return;
  if (act === 'accept') {
    e.stopPropagation();
    await togglePairLink(a, b);
    return;
  }
  if (act === 'inspect') {
    e.stopPropagation();
    app.highlightPair = row.dataset.key;
    openLinkInspector(row.dataset.key);
    return;
  }
  app.highlightPair = row.dataset.key;
  app.selected = null;
  renderPairs();
  renderScans();
  renderSelection();
  requestRender();
  $('#scan-list .scan.link-end')?.scrollIntoView({ block: 'nearest' });
});

$('#pair-filter').onchange = (e) => {
  app.pairFilter = e.target.value;
  renderPairs();
};

$('#group-chips').addEventListener('click', (e) => {
  const chip = e.target.closest('.group-chip');
  if (!chip) return;
  if (e.detail >= 2) {
    setReferenceForGroup(chip.dataset.gid);
    return;
  }
  focusGroup(chip.dataset.gid);
});

$('#selection').addEventListener('click', async (e) => {
  if (e.target.dataset.pair === 'inspect' && app.highlightPair) {
    openLinkInspector(app.highlightPair);
    return;
  }
  if (e.target.dataset.pair === 'measure' && app.highlightPair) {
    const [a, b] = app.highlightPair.split('|');
    try {
      const data = await postJSON('/api/pairs/accuracy', { a, b });
      const pair = app.state?.pairs.find((p) => p.a === a && p.b === b) || app.state?.pairs.find((p) => p.a === b && p.b === a);
      if (pair) {
        pair.accuracy = data.accuracy;
        pair.feature_matches = data.feature_matches;
      }
      await refreshState(true);
      renderSelection();
      toast('Surface deviation measured', 'ok');
    } catch (err) {
      toast(String(err), 'err');
    }
    return;
  }
  if (e.target.dataset.pair === 'toggle' && app.highlightPair) {
    const [a, b] = app.highlightPair.split('|');
    const pair = app.state.pairs.find((p) => p.a === a && p.b === b);
    if (pair) {
      await postJSON('/api/pairs/accept', { a, b, accepted: !pair.accepted });
      await refreshState(true);
      await runJob(postJSON('/api/solve', {}), 'Re-solving');
      app.highlightPair = `${a}|${b}`;
      renderSelection();
    }
    return;
  }
  const act = e.target.closest('[data-sel]')?.dataset.sel;
  if (!act) return;
  const sid = app.selected;
  if (!sid) return;
  if (act === 'place') {
    const g = groupOf(sid);
    startAssist(sid, { wholeGroup: !!(g && !g.unconnected && g.id !== 0 && g.members.length > 1) });
    return;
  }
  if (act === 'ref') {
    await runJob(postJSON('/api/reference', { id: sid }), 'Setting reference');
    return;
  }
  if (act === 'reset') {
    await applyScanTransform(sid, identity(), 'Scan pose reset');
    return;
  }
  if (act === 'recover') {
    try {
      const from = document.querySelector('#recover-from')?.value || null;
      const res = await postJSON(`/api/scans/${sid}/recover`, { from_scan: from });
      await refreshState(true);
      toast(`Pose recovered from ${(app.state.scans.find((s) => s.id === res.from) || {}).label || res.from}`, 'ok', {
        label: 'Undo',
        fn: async () => {
          await postJSON('/api/transform', { id: sid, matrix: res.previous });
          await refreshState(true);
        },
      });
    } catch (err) {
      toast(String(err), 'err');
    }
    return;
  }
  if (act === 'undo-transform') {
    if (app.transformUndo && app.transformUndo.sid === sid) {
      await applyScanTransform(sid, app.transformUndo.matrix, 'Pose restored');
      app.transformUndo = null;
    }
    return;
  }
  if (act === 'iso') {
    app.hidden = new Set(app.state.scans.filter((s) => s.id !== sid).map((s) => s.id));
    for (const [id, obj] of app.scans) {
      const vis = id === sid;
      if (obj?.points) obj.points.visible = vis;
      if (obj?.featureEdge) obj.featureEdge.visible = vis && app.showFeatures;
      if (obj?.featurePlane) obj.featurePlane.visible = vis && app.showFeatures;
    }
    renderScans();
    requestRender();
  }
  if (act === 'showall') {
    app.hidden.clear();
    app.focusedGroup = null;
    renderGroups();
    for (const s of app.scans.values()) {
      if (s.points) s.points.visible = true;
      if (s.featureEdge) s.featureEdge.visible = app.showFeatures;
      if (s.featurePlane) s.featurePlane.visible = app.showFeatures;
    }
    renderScans();
    requestRender();
  }
});

$('#selection').addEventListener('change', async (e) => {
  const field = e.target.dataset.tf;
  if (!field || !app.selected) return;
  const data = app.state.scans.find((s) => s.id === app.selected);
  if (!data) return;
  const m = mat4FromRow(data.transform);
  const t = new THREE.Vector3();
  const q = new THREE.Quaternion();
  m.decompose(t, q, new THREE.Vector3());
  const eu = new THREE.Euler().setFromQuaternion(q, 'XYZ');
  const v = parseFloat(e.target.value) || 0;
  if (field === 'x') t.x = v;
  if (field === 'y') t.y = v;
  if (field === 'z') t.z = v;
  if (field === 'rx') eu.x = rotRad(v);
  if (field === 'ry') eu.y = rotRad(v);
  if (field === 'rz') eu.z = rotRad(v);
  q.setFromEuler(eu);
  m.compose(t, q, new THREE.Vector3(1, 1, 1));
  const arr = m.clone().transpose().toArray();
  const rows = [[], [], [], []];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) rows[r][c] = arr[r * 4 + c];
  await applyScanTransform(app.selected, rows, 'Pose edited');
});

async function applyScanTransform(sid, matrix, label) {
  const data = app.state?.scans.find((s) => s.id === sid);
  if (!data) return;
  const prev = JSON.parse(JSON.stringify(data.transform));
  app.transformUndo = { sid, matrix: prev };
  await postJSON('/api/transform', { id: sid, matrix });
  await refreshState(true);
  toast(label, 'ok', {
    label: 'Undo',
    fn: async () => {
      await postJSON('/api/transform', { id: sid, matrix: prev });
      app.transformUndo = null;
      await refreshState(true);
      toast('Pose restored', 'ok');
    },
  });
}

function identity() {
  return [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
}

/* viewport toolbar */
$('#color-mode').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-mode]');
  if (!btn) return;
  app.colorMode = btn.dataset.mode;
  $$('#color-mode button').forEach((b) => b.classList.toggle('on', b === btn));
  applyPointUniforms(true);
  requestRender();
});

$('#toggle-features').onchange = (e) => {
  app.showFeatures = e.target.checked;
  applyFeatureVisibility();
};

$('#toggle-scanner-links').onchange = (e) => {
  canvas.style.cursor = '';
  app.showScannerLinks = e.target.checked;
  requestRender();
};

$('#toggle-before').onchange = (e) => {
  app.showBefore = e.target.checked;
  updateOffsets();
  fitAll();
};

$('#toggle-main').onchange = (e) => {
  app.mainOnly = e.target.checked;
  requestRender();
};

$('#toggle-grid').onchange = (e) => {
  app.layout = e.target.checked ? 'grid' : 'true';
  app.layoutUserSet = true;
  $('#spread').disabled = !e.target.checked;
  updateOffsets();
  fitAll();
};

$('#spread').oninput = (e) => {
  app.spread = parseFloat(e.target.value);
  updateOffsets();
  requestRender();
};

/* settings */
$('#inspector').addEventListener('input', (e) => {
  if (e.target.id && e.target.id.startsWith('p-')) {
    app.paramsDirty = true;
  }
});
$('#inspector').addEventListener('change', (e) => {
  if (e.target.id && e.target.id.startsWith('p-')) {
    app.paramsDirty = true;
  }
});

$('#p-accept').oninput = (e) => {
  $('#p-accept-out').textContent = parseFloat(e.target.value).toFixed(2);
};
$('#p-workers').oninput = (e) => {
  $('#p-workers-out').textContent = e.target.value;
};
$('#btn-apply-params').onclick = async () => {
  const body = {
    accept_score: parseFloat($('#p-accept').value),
    preview_points: parseInt($('#p-preview').value, 10),
    pair_workers: parseInt($('#p-workers').value, 10),
    use_pose: $('#p-pose').checked,
    detect_features: $('#p-features').checked,
    cleanup: $('#p-cleanup').value,
  };
  const res = await postJSON('/api/params', body);
  app.paramsDirty = false;
  app.reloadPending = !!res.reload_required;
  $('#reload-warning').classList.toggle('hidden', !app.reloadPending);
  const c = app.state?.cache;
  if (c) {
    $('#cache-status').textContent =
      `Noise removed: ${(c.noise_pct * 100).toFixed(1)}% of points`;
  }
  await refreshState(true);
  if (app.reloadPending) toast('Settings saved. Press Load scans to use them.', '');
};

/* ------------------------------------------------------------------ browser */

let browseData = null;
let browseSelected = new Set();

async function openBrowser(path = null) {
  browseSelected = new Set();
  $('#modal-files').classList.remove('hidden');
  await browse(path);
}

async function browse(path = null) {
  const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : '/api/browse';
  try {
    browseData = await getJSON(url);
  } catch (err) {
    toast(String(err), 'err');
    return;
  }
  $('#browse-path').value = browseData.path;
  $('#browse-up').disabled = !browseData.parent;
  renderBrowse();
}

function renderBrowse() {
  const host = $('#browse-list');
  if (!browseData) return;
  const rows = [];
  if (browseData.parent) {
    rows.push(`<div class="browse-row dir" data-dir="${browseData.parent}"><span class="icon">↑</span><span>..</span></div>`);
  }
  for (const d of browseData.dirs) {
    rows.push(`<div class="browse-row dir" data-dir="${d.path}"><span class="icon">▣</span><span>${d.name}</span></div>`);
  }
  for (const f of browseData.files) {
    const on = browseSelected.has(f.path);
    rows.push(`<div class="browse-row file" data-file="${f.path}">
      <input type="checkbox" ${on ? 'checked' : ''} data-file="${f.path}" />
      <span class="icon">◈</span><span>${f.name}</span><span class="size">${fmtSize(f.size)}</span>
    </div>`);
  }
  host.innerHTML = rows.join('') || '<div class="empty">No E57 files here.</div>';
  updateBrowseSummary();
}

function updateBrowseSummary() {
  $('#browse-summary').textContent = `${browseSelected.size} selected`;
  $('#browse-add').disabled = browseSelected.size === 0;
}

$('#browse-list').addEventListener('click', (e) => {
  const dir = e.target.closest('[data-dir]');
  if (dir) { browse(dir.dataset.dir); return; }
  const fileRow = e.target.closest('[data-file]');
  if (!fileRow) return;
  const path = fileRow.dataset.file;
  const cb = fileRow.querySelector('input[type=checkbox]');
  if (e.target !== cb) cb.checked = !cb.checked;
  if (cb.checked) browseSelected.add(path);
  else browseSelected.delete(path);
  updateBrowseSummary();
});

$('#browse-up').onclick = () => { if (browseData?.parent) browse(browseData.parent); };
$('#browse-go').onclick = () => browse($('#browse-path').value.trim());
$('#browse-path').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') browse($('#browse-path').value.trim());
});
$('#browse-add').onclick = async () => {
  if (!browseSelected.size) return;
  const paths = [...browseSelected];
  const force = $('#browse-force').checked;
  $('#modal-files').classList.add('hidden');
  await addScans(paths, force);
};
$('#browse-add-all').onclick = async () => {
  if (!browseData) return;
  const force = $('#browse-force').checked;
  $('#modal-files').classList.add('hidden');
  await addScans([browseData.path], force);
};

/* ------------------------------------------------------------------ export */

function openExport() {
  const st = app.state;
  if (!st) return;
  $('#modal-export').classList.remove('hidden');
  const active = st.groups?.[0]?.members?.length || st.scans.filter((s) => s.loaded).length;
  const all = st.scans.filter((s) => s.loaded).length;
  updateExportSummary();
  if (!$('#ex-path').value) {
    const s = st.scans.find((x) => x.loaded);
    const dir = s ? s.file.replace(/\/[^/]+$/, '') : '.';
    $('#ex-path').value = `${dir}/registered_merged.e57`;
  }
}

function updateExportSummary() {
  const st = app.state;
  if (!st) return;
  const groups = st.groups || [];
  const main = groups[0]?.members?.length || st.scans.filter((s) => s.loaded).length;
  const all = st.scans.filter((s) => s.loaded).length;
  const scope = $('#ex-scope').value;
  const fmt = $('#ex-format').value;
  const isMesh = fmt.startsWith('mesh-');
  let txt = isMesh
    ? `mesh · main group: ${main} scan(s) · triangles from a decimated merged cloud`
    : `main group: ${main} scan(s) · all: ${all} scan(s)`;
  if (scope === 'all' && groups.length > 1) {
    txt += ` · ${groups.length} groups will be written as separate scans`;
    if (fmt === 'e57') {
      $('#ex-mode').value = 'separate';
      $('#ex-mode').disabled = true;
    }
  } else if (fmt === 'e57') {
    $('#ex-mode').disabled = false;
  }
  if (scope === 'all' && fmt !== 'e57' && !isMesh) {
    txt += ' · text/PLY export merges all groups into one coordinate list';
  }
  $('#ex-mesh-controls').classList.toggle('hidden', !isMesh);
  if (isMesh) {
    $('#ex-mode').value = 'merged';
    $('#ex-mode').disabled = true;
    $('#ex-precision').disabled = true;
  }
  $('#export-summary').textContent = txt;
}

$('#ex-scope').onchange = updateExportSummary;

$('#ex-format').onchange = () => {
  const fmt = $('#ex-format').value;
  const ext = fmt === 'mesh-ply' ? 'ply' : fmt === 'mesh-obj' ? 'obj' : fmt === 'mesh-glb' ? 'glb' : fmt;
  $('#ex-path').value = $('#ex-path').value.replace(/\.(e57|ply|pts|xyz|obj|glb)$/i, `.${ext}`);
  $('#ex-precision').disabled = fmt !== 'e57';
  $('#ex-mode').disabled = fmt !== 'e57' && !fmt.startsWith('mesh-');
  updateExportSummary();
};
function updateExportDensity() {
  const mode = $('#ex-density').value;
  $('#ex-density-controls').classList.toggle('hidden', mode === 'full');
  $('#ex-cap').disabled = mode !== 'adaptive';
  $('#ex-density-hint').textContent =
    mode === 'full'
      ? 'Exports every valid point (minus filtered noise).'
      : mode === 'adaptive'
        ? 'Keeps all points in sparse areas; only voxels with more than the cap are thinned.'
        : 'Keeps one representative point per voxel in every area.';
}

$('#ex-voxel').oninput = () => {
  const v = parseFloat($('#ex-voxel').value);
  $('#ex-voxel-out').textContent = v > 0 ? `${(v * 100).toFixed(1)} cm` : 'full resolution';
};
$('#ex-density').onchange = updateExportDensity;
$('#export-run').onclick = async () => {
  const included = app.state.scans.filter((s) => s.loaded && s.included !== false).map((s) => s.id);
  const densityMode = $('#ex-density').value;
  const densityVoxel = parseFloat($('#ex-voxel').value) || 0.02;
  const fmt = $('#ex-format').value;
  if (fmt.startsWith('mesh-')) {
    const body = {
      path: $('#ex-path').value.trim(),
      scope: $('#ex-scope').value,
      ids: $('#ex-scope').value === 'selected' ? included : null,
      method: $('#ex-mesh-method').value,
      voxel: densityVoxel,
      depth: parseInt($('#ex-mesh-depth').value, 10) || 9,
      colors: $('#ex-colors').checked,
      intensity: $('#ex-intensity').checked,
      intensity_ramp: fmt !== 'mesh-ply',
    };
    if (!body.path) { toast('Enter an output path', 'err'); return; }
    $('#export-status').textContent = 'building mesh…';
    $('#modal-export').classList.add('hidden');
    const res = await runJob(postJSON('/api/export-mesh', body), 'Building mesh');
    if (res) {
      app.lastMeshPath = res.path;
      toast(`Mesh: ${fmtInt(res.vertices || 0)} vertices · ${fmtInt(res.triangles || 0)} triangles → ${res.path}`, 'ok');
      await loadMesh(res.path);
    }
    return;
  }
  const body = {
    path: $('#ex-path').value.trim(),
    format: $('#ex-format').value,
    mode: $('#ex-mode').value,
    scope: $('#ex-scope').value,
    ids: $('#ex-scope').value === 'selected' ? included : null,
    voxel: densityMode === 'voxel' ? densityVoxel : 0,
    density_cap: densityMode === 'adaptive' ? (parseInt($('#ex-cap').value, 10) || 3) : 0,
    density_voxel: densityVoxel,
    filter_noise: $('#ex-filter-noise').checked,
    precision: $('#ex-precision').value,
    colors: $('#ex-colors').checked,
    intensity: $('#ex-intensity').checked,
    name: 'registered',
  };
  if (!body.path) { toast('Enter an output path', 'err'); return; }
  $('#export-status').textContent = 'writing…';
  $('#modal-export').classList.add('hidden');
  const res = await runJob(postJSON('/api/export', body), 'Exporting');
  if (res) {
    toast(`Wrote ${fmtSize(res.bytes)} → ${res.path}`, 'ok');
  }
};

/* ------------------------------------------------------------------ modal plumbing */

$$('[data-close]').forEach((b) => {
  b.onclick = () => b.closest('.modal-backdrop').classList.add('hidden');
});
$$('.modal-backdrop').forEach((m) => {
  m.addEventListener('click', (e) => {
    if (e.target === m) m.classList.add('hidden');
  });
});

// drag & drop hint: paths cannot be dropped from the OS without upload, so
// offer the path of the dropped folder when the browser exposes it.
window.addEventListener('dragover', (e) => { e.preventDefault(); });
window.addEventListener('drop', (e) => {
  e.preventDefault();
  toast('Use “Add scans…” to pick a folder on disk.', '');
});

/* ------------------------------------------------------------------ link inspector */

const link = {
  key: null,
  renderer: null,
  scene: null,
  camTop: null,
  camSide: null,
  points: [],
  markers: null,
  lines: null,
  planes: [],
  bounds: null,
  axis: 'y',
  pos: 0.5,
  thick: 0.2,
  viewTop: { cx: 0, cy: 0, half: 5 },
  viewSide: { cx: 0, cz: 0, half: 5 },
  data: null,
  raf: 0,
  virtualPair: null,
  scanIds: [],
  pair: null,
  playing: false,
  dir: 1,
  speed: 0.1,
  lastT: 0,
  scrubRAF: 0,
};

function linkPairInfo() {
  return link.pair || link.virtualPair || null;
}

function scansWorldBBox(ids) {
  let bb = null;
  for (const id of ids) {
    const s = app.scans.get(id);
    if (!s || !s.data.bbox) continue;
    const local = s.data.bbox;
    const M = scanDisplayMatrix(s);
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(
        i & 1 ? local[1][0] : local[0][0],
        i & 2 ? local[1][1] : local[0][1],
        i & 4 ? local[1][2] : local[0][2]
      ).applyMatrix4(M);
      if (!bb) bb = new THREE.Box3(p.clone(), p.clone());
      else bb.expandByPoint(p);
    }
  }
  return bb;
}

function setupLinkRenderer() {
  if (link.renderer) return;
  const canvas = $('#link-canvas');
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.setClearColor(0x090b11, 1);
  renderer.setScissorTest(true);
  renderer.localClippingEnabled = true;
  link.renderer = renderer;
  link.scene = new THREE.Scene();
  link.scene.background = new THREE.Color(0x090b11);
  link.camTop = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1e7);
  link.camSide = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 1e7);
}

function disposeLinkScene() {
  if (!link.scene) return;
  for (const obj of [link.markers, link.lines]) {
    if (!obj) continue;
    link.scene.remove(obj);
    obj.geometry?.dispose();
    obj.material?.dispose?.();
  }
  link.markers = link.lines = null;
  for (const obj of link.points) {
    link.scene.remove(obj);
    obj.material.dispose();
  }
  link.points = [];
}

function buildLinkScene(pair = null) {
  setupLinkRenderer();
  disposeLinkScene();
  const ids = link.scanIds.length ? link.scanIds : link.key.split('|');
  link.pair = pair;
  link.bounds = scansWorldBBox(ids);
  if (!link.bounds) return;
  const voxel = app.state.params.voxel || 0.05;

  link.planes = [
    new THREE.Plane(new THREE.Vector3(0, 1, 0), -1),
    new THREE.Plane(new THREE.Vector3(0, -1, 0), -1),
  ];

  for (const id of ids) {
    const scan = app.scans.get(id);
    if (!scan || !scan.geometry) continue;
    const material = new THREE.PointsMaterial({
      color: new THREE.Color(scan.data.color),
      size: 2.8,
      sizeAttenuation: false,
      clippingPlanes: link.planes,
      depthWrite: true,
    });
    const obj = new THREE.Points(scan.geometry, material);
    obj.matrixAutoUpdate = false;
    obj.matrix.copy(scanDisplayMatrix(scan));
    obj.matrixWorldNeedsUpdate = true;
    link.scene.add(obj);
    link.points.push(obj);
  }

  // matched significant features are meaningful for a pair; for 3+ scans the
  // colour-separated point clouds are the validation surface.
  const matches = pair?.feature_matches?.edge_matches || [];
  if (matches.length && pair) {
    const sphereGeo = new THREE.SphereGeometry(Math.max(0.035, voxel * 1.4), 10, 7);
    const markerGroup = new THREE.Group();
    const linePos = [];
    const lineCol = [];
    const good = new THREE.Color(0x6bcb77);
    const warn = new THREE.Color(0xffd93d);
    const bad = new THREE.Color(0xff6b6b);
    const scanA = app.scans.get(pair.a);
    const scanB = app.scans.get(pair.b);
    const MA = scanA ? scanDisplayMatrix(scanA) : new THREE.Matrix4();
    const MB = scanB ? scanDisplayMatrix(scanB) : new THREE.Matrix4();
    for (const m of matches) {
      if (!m.a_centroid || !m.b_centroid) continue;
      const pa = new THREE.Vector3().fromArray(m.a_centroid).applyMatrix4(MA);
      const pb = new THREE.Vector3().fromArray(m.b_centroid).applyMatrix4(MB);
      const dmm = m.dist_mm ?? pa.distanceTo(pb) * 1000;
      const col = dmm < 10 ? good : dmm < 30 ? warn : bad;
      const ma = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: col, clippingPlanes: link.planes }));
      ma.position.copy(pa);
      const mb = new THREE.Mesh(sphereGeo, new THREE.MeshBasicMaterial({ color: col.clone().multiplyScalar(0.65), clippingPlanes: link.planes }));
      mb.position.copy(pb);
      markerGroup.add(ma, mb);
      linePos.push(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z);
      lineCol.push(col.r, col.g, col.b, col.r, col.g, col.b);
    }
    link.markers = markerGroup;
    link.scene.add(markerGroup);

    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(linePos, 3));
    lg.setAttribute('color', new THREE.Float32BufferAttribute(lineCol, 3));
    link.lines = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ vertexColors: true, clippingPlanes: link.planes }));
    link.scene.add(link.lines);
  }

  link.axis = $('#link-axis button.on')?.dataset.axis || 'y';
  const pts = scansPointsSample(ids);
  centerCutOnCloud(pts);
  updateLinkCut();
  fitLinkViews();
}

function updateLinkCut() {
  if (!link.bounds || !link.planes.length) return;
  const lo = link.bounds.min.clone();
  const hi = link.bounds.max.clone();
  const axis = link.axis;
  const min = axis === 'x' ? lo.x : axis === 'y' ? lo.y : lo.z;
  const max = axis === 'x' ? hi.x : axis === 'y' ? hi.y : hi.z;
  const c = min + (max - min) * link.pos;
  const half = Math.max(link.thick / 2, 0.005);
  const n = new THREE.Vector3(axis === 'x' ? 1 : 0, axis === 'y' ? 1 : 0, axis === 'z' ? 1 : 0);
  link.planes[0].normal.copy(n);
  link.planes[0].constant = -(c - half);
  link.planes[1].normal.copy(n).negate();
  link.planes[1].constant = c + half;
  $('#link-pos-out').textContent = `${(link.pos * 100).toFixed(0)}%`;
  $('#link-thick-out').textContent = `${(link.thick * 100).toFixed(0)} cm`;
  requestLinkRender();
}

function scansPointsSample(ids, maxEach = 35000) {
  const out = [];
  for (const id of ids) {
    const scan = app.scans.get(id);
    if (!scan || !scan.points) continue;
    const arr = scan.points.geometry.attributes.position.array;
    const n = arr.length / 3;
    if (!n) continue;
    const step = Math.max(1, Math.floor(n / maxEach));
    const M = scanDisplayMatrix(scan);
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i += step) {
      v.fromArray(arr, i * 3).applyMatrix4(M);
      out.push(v.clone());
    }
  }
  return out;
}

function linkAxisValue(p) {
  return link.axis === 'x' ? p.x : link.axis === 'y' ? p.y : p.z;
}

function linkSlabBounds(points = null) {
  if (!link.bounds) return null;
  const pts = points || scansPointsSample(link.scanIds);
  if (!pts.length) return null;
  const lo = link.bounds.min.clone();
  const hi = link.bounds.max.clone();
  const min = link.axis === 'x' ? lo.x : link.axis === 'y' ? lo.y : lo.z;
  const max = link.axis === 'x' ? hi.x : link.axis === 'y' ? hi.y : hi.z;
  const c = min + (max - min) * link.pos;
  const half = Math.max(link.thick / 2, 0.01);
  let bb = null;
  for (const p of pts) {
    const v = linkAxisValue(p);
    if (v < c - half || v > c + half) continue;
    if (!bb) bb = new THREE.Box3(p.clone(), p.clone());
    else bb.expandByPoint(p);
  }
  return bb;
}

function centerCutOnCloud(points = null) {
  if (!link.bounds) return;
  const pts = points || scansPointsSample(link.scanIds);
  if (!pts.length) return;
  const vals = pts.map(linkAxisValue).sort((x, y) => x - y);
  const med = vals[Math.floor(vals.length / 2)];
  const lo = link.bounds.min.clone();
  const hi = link.bounds.max.clone();
  const min = link.axis === 'x' ? lo.x : link.axis === 'y' ? lo.y : lo.z;
  const max = link.axis === 'x' ? hi.x : link.axis === 'y' ? hi.y : hi.z;
  if (max - min < 1e-6) return;
  link.pos = clamp((med - min) / (max - min), 0.03, 0.97);
  $('#link-pos').value = Math.round(link.pos * 1000);
}

function fitLinkViews() {
  if (!link.bounds) return;
  const slab = linkSlabBounds();
  const bb = slab && !slab.isEmpty() ? slab : link.bounds;
  const cb = bb.getCenter(new THREE.Vector3());
  const sz = bb.getSize(new THREE.Vector3());
  link.viewTop = { cx: cb.x, cy: cb.y, half: Math.max(sz.x, sz.y, 0.5) * 0.62 };
  link.viewSide = { cx: cb.x, cz: cb.z, half: Math.max(sz.x, sz.z, 0.5) * 0.62 };
  requestLinkRender();
}

function requestLinkRender() {
  if (link.raf) return;
  link.raf = requestAnimationFrame(() => {
    link.raf = 0;
    renderLink();
  });
}

function renderLink() {
  if (!link.renderer || $('#modal-link').classList.contains('hidden')) return;
  const canvas = $('#link-canvas');
  const w = Math.max(2, canvas.clientWidth);
  const h = Math.max(2, canvas.clientHeight);
  const r = link.renderer;
  r.setSize(w, h, false);
  const halfW = Math.floor(w / 2);

  const renderPane = (cam, view, x, pw, orthoAxis) => {
    const aspect = pw / h;
    let halfH = view.half;
    if (aspect < 1) halfH = view.half / aspect;
    const hw = halfH * aspect;
    cam.left = -hw; cam.right = hw; cam.top = halfH; cam.bottom = -halfH;
    cam.near = 0.05; cam.far = 1e7; cam.updateProjectionMatrix();
    if (orthoAxis === 'top') {
      cam.up.set(0, 1, 0);
      cam.position.set(view.cx, view.cy, 1e5);
      cam.lookAt(view.cx, view.cy, 0);
    } else {
      cam.up.set(0, 0, 1);
      cam.position.set(view.cx, -1e5, view.cz);
      cam.lookAt(view.cx, 0, view.cz);
    }
    r.setViewport(x, 0, pw, h);
    r.setScissor(x, 0, pw, h);
    r.render(link.scene, cam);
  };
  renderPane(link.camTop, link.viewTop, 0, halfW, 'top');
  renderPane(link.camSide, link.viewSide, halfW, w - halfW, 'side');
}

function renderLinkStats(pair, data) {
  if (!pair) {
    const ids = link.scanIds;
    const st = app.state;
    const rows = ids.map((id) => {
      const scan = st.scans.find((s) => s.id === id);
      const label = scan?.label || id;
      const group = scan && scan.group >= 0 ? String.fromCharCode(65 + scan.group) : '—';
      return `<div class="link-legend-row"><span class="swatch" style="background:${scan?.color || '#888'}"></span><span>${escapeHtml(label)}</span><span class="muted">group ${group}</span></div>`;
    });
    const sel = new Set(ids);
    const links = (st.pairs || [])
      .filter((p) => sel.has(p.a) && sel.has(p.b) && p.accepted)
      .sort((a, b) => (b.score ?? 0) - (a.score ?? 0));
    const linkRows = links.slice(0, 12).map((p) => {
      const la = st.scans.find((s) => s.id === p.a)?.label || p.a;
      const lb = st.scans.find((s) => s.id === p.b)?.label || p.b;
      const mm = p.accuracy && p.accuracy.mean_mm != null ? ` · ${Number(p.accuracy.mean_mm).toFixed(1)} mm` : '';
      return `<div class="link-legend-row"><span>${escapeHtml(la)} ↔ ${escapeHtml(lb)}</span><span>${((p.score ?? 0) * 100).toFixed(0)}%${mm}</span></div>`;
    });
    $('#link-stats').innerHTML = `
      <div class="big">${ids.length} scans</div>
      <div class="small">cross-section · each scan keeps its own colour</div>
      <div class="link-legend">${rows.join('')}</div>
      <div class="kv"><span>links between selection</span><span>${links.length}</span></div>
      <div class="link-legend">${linkRows.join('') || '<div class="link-legend-row"><span>no accepted links between these scans</span><span>—</span></div>'}</div>`;
    $('#link-feature-list').innerHTML = '';
    return;
  }
  const acc = (data && data.accuracy) || pair.accuracy || {};
  const fm = (data && data.feature_matches) || pair.feature_matches || {};
  const mm = (v) => (v == null ? '—' : `${Number(v).toFixed(1)} mm`);
  const pct = (v) => (v == null ? '—' : `${(Number(v) * 100).toFixed(0)}%`);
  const [a, b] = link.key.split('|');
  const la = app.state.scans.find((s) => s.id === a)?.label || a;
  const lb = app.state.scans.find((s) => s.id === b)?.label || b;
  $('#link-title').textContent = `${la} ↔ ${lb}`;
  $('#link-stats').innerHTML = `
    <div class="big">${mm(acc.mean_mm)}</div>
    <div class="small">mean surface deviation${acc.n ? ` · ${acc.n.toLocaleString()} points` : ''}</div>
    <div class="kv">
      <span>median</span><span>${mm(acc.median_mm)}</span>
      <span>p95</span><span>${mm(acc.p95_mm)}</span>
      <span>rms</span><span>${mm(acc.rms_mm)}</span>
      <span>inliers</span><span>${pct(acc.inlier_ratio)}</span>
      <span>link score</span><span>${pct(pair.score)}</span>
      <span>structure</span><span>${pct(pair.structure_overlap)}</span>
      <span>edge clusters</span><span>${fm.edges_matched ?? 0}/${fm.edges_total ?? 0} significant</span>
      <span>edge centroids</span><span>${fm.edges_accuracy ? mm(fm.edges_accuracy.mean_mm) : '—'}${fm.edges_tight != null ? ` (${fm.edges_tight} tight)` : ''}</span>
      <span>planes</span><span>${fm.planes_matched ?? 0}/${fm.planes_total ?? 0} significant</span>
      <span>plane area</span><span>${fm.planes_area_total ? `${Math.round((fm.planes_area_matched / fm.planes_area_total) * 100)}%` : '—'}</span>
    </div>`;
  const list = (fm.edge_matches || []).slice().sort((x, y) => (x.dist_mm ?? 0) - (y.dist_mm ?? 0)).slice(0, 18);
  $('#link-feature-list').innerHTML = list.length
    ? list
        .map((m, i) => {
          const d = m.dist_mm ?? 0;
          const cls = d < 10 ? 'good' : d < 30 ? 'warn' : 'bad';
          return `<div><span>edge ${i + 1}</span><span class="${cls}">${d.toFixed(1)} mm</span></div>`;
        })
    .join('')
    : '<div><span>no matched edge clusters</span><span>—</span></div>';
}

async function openLinkInspector(idsOrKey) {
  if (!idsOrKey) return;
  const ids = Array.isArray(idsOrKey) ? idsOrKey.slice() : String(idsOrKey).split('|');
  if (ids.length < 2) return;
  link.scanIds = ids;
  link.key = ids.join('|');
  let pair = null;
  if (ids.length === 2) {
    const [ka, kb] = ids;
    pair = app.state?.pairs.find(
      (p) => (p.a === ka && p.b === kb) || (p.a === kb && p.b === ka)
    );
    link.virtualPair = pair
      ? null
      : {
          a: ka,
          b: kb,
          score: null,
          structure_overlap: null,
          overlap: null,
          rmse: null,
          accepted: false,
          virtual: true,
          feature_matches: {},
        };
    pair = pair || link.virtualPair;
  } else {
    link.virtualPair = null;
  }
  link.pair = pair;
  const labels = ids.map((id) => app.state?.scans.find((s) => s.id === id)?.label || id);
  $('#link-title').textContent =
    ids.length === 2 ? `${labels[0]} ↔ ${labels[1]}` : `Cut-through · ${ids.length} scans`;
  stopScrub();
  $('#modal-link').classList.remove('hidden');
  $('#link-register').classList.toggle('hidden', !(pair && pair.virtual));
  $('#link-reject').classList.toggle('hidden', !(pair && !pair.virtual));
  $('#link-recompute').classList.toggle('hidden', ids.length !== 2);
  buildLinkScene(pair);
  renderLinkStats(pair, null);
  requestLinkRender();
  if (!pair) return;
  try {
    $('#link-stats').insertAdjacentHTML('beforeend', '<div class="hint" id="link-loading">computing surface deviation…</div>');
    const data = await postJSON('/api/pairs/accuracy', { a: pair.a, b: pair.b });
    if (link.key !== ids.join('|')) return;
    link.data = data;
    pair.feature_matches = data.feature_matches || pair.feature_matches || {};
    renderLinkStats(pair, data);
    buildLinkScene(pair);
  } catch (err) {
    $('#link-loading')?.remove();
  }
}

$('#link-axis').addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-axis]');
  if (!btn) return;
  link.axis = btn.dataset.axis;
  $$('#link-axis button').forEach((b) => b.classList.toggle('on', b === btn));
  const pts = scansPointsSample(link.scanIds);
  centerCutOnCloud(pts);
  updateLinkCut();
  fitLinkViews();
});

$('#link-pos').oninput = (e) => {
  link.pos = parseFloat(e.target.value) / 1000;
  updateLinkCut();
};
$('#link-thick').oninput = (e) => {
  link.thick = Math.max(0.02, parseFloat(e.target.value) / 100);
  updateLinkCut();
};
$('#link-features').onchange = (e) => {
  if (link.markers) link.markers.visible = e.target.checked;
  requestLinkRender();
};
$('#link-lines').onchange = (e) => {
  if (link.lines) link.lines.visible = e.target.checked;
  requestLinkRender();
};
$('#link-fit').onclick = fitLinkViews;
$('#link-recompute').onclick = async () => {
  const pair = linkPairInfo();
  if (!pair) return;
  $('#link-stats').insertAdjacentHTML('beforeend', '<div class="hint" id="link-loading">recomputing…</div>');
  try {
    const data = await postJSON('/api/pairs/accuracy', { a: pair.a, b: pair.b });
    link.data = data;
    if (data.feature_matches) {
      pair.feature_matches = data.feature_matches;
      buildLinkScene(pair);
    }
    renderLinkStats(pair, data);
  } catch (err) {
    toast(String(err), 'err');
    $('#link-loading')?.remove();
  }
};
$('#link-register').onclick = async () => {
  const pair = linkPairInfo();
  if (!pair) return;
  const a = pair.a, b = pair.b;
  $('#link-register').disabled = true;
  try {
    const res = await postJSON('/api/register', { pairs: [[a, b]] });
    await runJob(Promise.resolve({ job: res.job }), 'Re-matching pair (all heavy matchers)');
    const real = app.state?.pairs.find(
      (p) => (p.a === a && p.b === b) || (p.a === b && p.b === a)
    );
    if (real) {
      link.virtualPair = null;
      link.key = `${real.a}|${real.b}`;
      $('#link-register').classList.add('hidden');
      $('#link-reject').classList.remove('hidden');
      buildLinkScene(real);
      renderLinkStats(real, null);
      const data = await postJSON('/api/pairs/accuracy', { a: real.a, b: real.b });
      real.feature_matches = data.feature_matches || {};
      buildLinkScene(real);
      renderLinkStats(real, data);
    }
  } catch (err) {
    toast(String(err), 'err');
  } finally {
    $('#link-register').disabled = false;
  }
};

$('#link-reject').onclick = async () => {
  const pair = linkPairInfo();
  if (!pair) return;
  await postJSON('/api/pairs/accept', { a: pair.a, b: pair.b, accepted: false });
  $('#modal-link').classList.add('hidden');
  await runJob(postJSON('/api/solve', {}), 'Re-solving');
  app.highlightPair = `${pair.a}|${pair.b}`;
  renderSelection();
};

function toggleScrub() {
  if (link.playing) stopScrub();
  else startScrub();
}

function startScrub() {
  if ($('#modal-link').classList.contains('hidden')) return;
  link.playing = true;
  link.dir = link.pos >= 0.999 ? -1 : link.pos <= 0.001 ? 1 : link.dir;
  link.lastT = performance.now();
  $('#link-scrub').textContent = '⏸ pause';
  $('#link-scrub').classList.add('primary');
  const step = (now) => {
    if (!link.playing) return;
    const dt = Math.min(0.1, (now - link.lastT) / 1000);
    link.lastT = now;
    link.pos += link.dir * link.speed * dt;
    if (link.pos >= 1) { link.pos = 1; link.dir = -1; }
    if (link.pos <= 0) { link.pos = 0; link.dir = 1; }
    $('#link-pos').value = Math.round(link.pos * 1000);
    updateLinkCut();
    link.scrubRAF = requestAnimationFrame(step);
  };
  link.scrubRAF = requestAnimationFrame(step);
}

function stopScrub() {
  link.playing = false;
  cancelAnimationFrame(link.scrubRAF);
  link.scrubRAF = 0;
  const b = $('#link-scrub');
  if (b) { b.textContent = '▶ scrub'; b.classList.remove('primary'); }
}

function stepCut(delta) {
  link.pos = clamp(link.pos + delta, 0, 1);
  $('#link-pos').value = Math.round(link.pos * 1000);
  updateLinkCut();
}

$('#link-scrub').onclick = toggleScrub;
$('#link-speed').oninput = (e) => {
  link.speed = parseFloat(e.target.value) / 100;
};
$('#link-slice').onclick = () => {
  link.thick = 0.02;
  $('#link-thick').value = 2;
  updateLinkCut();
};
$('#link-mark').onclick = async () => {
  if (!link.bounds) return;
  const lo = link.bounds.min, hi = link.bounds.max;
  const full = hi.clone().sub(lo);
  const c = lo.clone().add(hi).multiplyScalar(0.5);
  if (link.axis === 'x') c.x = lo.x + full.x * link.pos;
  else if (link.axis === 'y') c.y = lo.y + full.y * link.pos;
  else c.z = lo.z + full.z * link.pos;
  const half = Math.max(link.thick / 2, 0.01);
  const size = [
    link.axis === 'x' ? half : Math.max(full.x / 2, 0.01),
    link.axis === 'y' ? half : Math.max(full.y / 2, 0.01),
    link.axis === 'z' ? half : Math.max(full.z / 2, 0.01),
  ];
  try {
    await postJSON('/api/cleanup/add', {
      kind: 'box', center: [c.x, c.y, c.z], size, op: 'delete',
      scans: link.scanIds && link.scanIds.length ? link.scanIds : null,
      label: 'cut slab',
    });
    await refreshState(true);
    toast('Slab removed from the preview and exports', 'ok');
  } catch (err) {
    toast(String(err), 'err');
  }
};

// stop the animation when the modal closes
$('#modal-link').querySelectorAll('[data-close]').forEach((b) => {
  b.addEventListener('click', stopScrub);
});
$('#modal-link').addEventListener('click', (e) => {
  if (e.target === $('#modal-link')) stopScrub();
});

// keyboard: arrows step, space plays/pauses, while the inspector is open
window.addEventListener('keydown', (e) => {
  if ($('#modal-link').classList.contains('hidden')) return;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement?.tagName)) return;
  if (e.key === ' ') { e.preventDefault(); toggleScrub(); }
  if (e.key === 'ArrowRight' || e.key === 'ArrowUp') { e.preventDefault(); stepCut(0.02); }
  if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') { e.preventDefault(); stepCut(-0.02); }
});

// pan / zoom in the link views
let linkDrag = null;
$('#link-canvas').addEventListener('pointerdown', (e) => {
  const r = e.target.getBoundingClientRect();
  linkDrag = { x: e.clientX, y: e.clientY, pane: e.clientX - r.left < r.width / 2 ? 'top' : 'side' };
  e.target.setPointerCapture(e.pointerId);
});
$('#link-canvas').addEventListener('pointermove', (e) => {
  if (!linkDrag) return;
  const r = e.target.getBoundingClientRect();
  const pw = r.width / 2;
  const h = Math.max(1, r.height);
  const dx = e.clientX - linkDrag.x;
  const dy = e.clientY - linkDrag.y;
  linkDrag.x = e.clientX;
  linkDrag.y = e.clientY;
  if (linkDrag.pane === 'top') {
    const wpp = (2 * link.viewTop.half) / h;
    link.viewTop.cx -= dx * wpp;
    link.viewTop.cy += dy * wpp;
  } else {
    const wpp = (2 * link.viewSide.half) / h;
    link.viewSide.cx -= dx * wpp;
    link.viewSide.cz += dy * wpp;
  }
  requestLinkRender();
});
$('#link-canvas').addEventListener('pointerup', () => { linkDrag = null; });
$('#link-canvas').addEventListener('wheel', (e) => {
  e.preventDefault();
  const r = e.target.getBoundingClientRect();
  const top = e.clientX - r.left < r.width / 2;
  const view = top ? link.viewTop : link.viewSide;
  view.half = clamp(view.half * Math.exp(e.deltaY * 0.0012), 0.05, 1e6);
  requestLinkRender();
}, { passive: false });
$('#link-canvas').addEventListener('dblclick', fitLinkViews);

// Keep an open page in sync with jobs/changes started outside it (CLI,
// API) without disturbing a modal or an input being edited.
setInterval(async () => {
  if (app.job) return;
  if (document.querySelector('.modal-backdrop:not(.hidden)')) return;
  if (document.activeElement?.matches('input, select, textarea')) return;
  try { await refreshState(true); } catch {}
}, 8000);

/* ------------------------------------------------------------ manual assist */
//
// "Place manually": drag a scan (or its whole group) roughly into place over
// another group in the top view, press Snap, and Accept.  Everything happens
// in the target group's frame; nothing is sent to the server until Snap.

const ASSIST_MOVING = '#ff9f45';
const ASSIST_TARGET = '#5fb3d9';
const assistGizmo = new THREE.Group();
assistGizmo.matrixAutoUpdate = false;
assistGizmo.visible = false;
scene.add(assistGizmo);

function errText(err) {
  const s = String(err?.message || err);
  const m = s.match(/\{.*\}/s);
  if (m) {
    try { return JSON.parse(m[0]).detail || s; } catch {}
  }
  return s.replace(/^Error:\s*/, '');
}

function matRows(m) {
  const a = m.clone().transpose().toArray();
  return [0, 1, 2, 3].map((r) => a.slice(r * 4, r * 4 + 4));
}

function scanLabel(sid) {
  return app.state?.scans.find((s) => s.id === sid)?.label || sid;
}

function groupOf(sid) {
  return (app.state?.groups || []).find((g) => g.members.includes(sid)) || null;
}

function localCenter(sid) {
  const bb = app.scans.get(sid)?.data.bbox;
  if (!bb) return new THREE.Vector3();
  return new THREE.Vector3((bb[0][0] + bb[1][0]) / 2, (bb[0][1] + bb[1][1]) / 2, (bb[0][2] + bb[1][2]) / 2);
}

// sample registration-frame points of `ids` (optionally with a pose override)
function regSample(ids, poseOf, maxEach = 4000) {
  const out = [];
  const v = new THREE.Vector3();
  for (const id of ids) {
    const s = app.scans.get(id);
    if (!s?.points) continue;
    const arr = s.points.geometry.attributes.position.array;
    const n = arr.length / 3;
    const step = Math.max(1, Math.floor(n / maxEach));
    const M = poseOf(id);
    for (let i = 0; i < n; i += step) out.push(v.fromArray(arr, i * 3).applyMatrix4(M).clone());
  }
  return out;
}

function quantile(vals, q) {
  if (!vals.length) return 0;
  const a = [...vals].sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.max(0, Math.floor((a.length - 1) * q)))];
}

function assistDelta() {
  const A = app.assist;
  return A.pose.clone().multiply(A.base.get(A.anchor).clone().invert());
}

function assistDisplayMatrix(sid) {
  const A = app.assist;
  return A.offset.clone().multiply(assistDelta()).multiply(A.base.get(sid));
}

// world (registration-frame) centre of the moving set, used as rotation pivot
function assistPivot() {
  const A = app.assist;
  return localCenter(A.anchor).applyMatrix4(A.pose);
}

function assistTargetChoices(moving) {
  const groups = app.state?.groups || [];
  const mov = new Set(moving);
  return groups.filter((g) => !g.members.every((m) => mov.has(m)) && g.members.some((m) => !mov.has(m)));
}

function startAssist(sid, { wholeGroup = true, targetGid = null } = {}) {
  const st = app.state;
  if (!st) return;
  const data = st.scans.find((s) => s.id === sid);
  if (!data?.loaded || !app.scans.get(sid)?.points) {
    toast('Load this scan first (press Load scans)', 'err');
    return;
  }
  if (app.assist) stopAssist(false);
  const own = groupOf(sid);
  const moving = wholeGroup && own && !own.unconnected ? [...own.members] : [sid];
  const choices = assistTargetChoices(moving);
  if (!choices.length) {
    toast('There is no other group to place it into', 'err');
    return;
  }
  const target =
    choices.find((g) => String(g.id) === String(targetGid)) ||
    choices.find((g) => g.id === 0 && !g.members.includes(sid)) ||
    [...choices].sort((a, b) => b.members.length - a.members.length)[0];
  const base = new Map();
  for (const m of moving) base.set(m, mat4FromRow(st.scans.find((s) => s.id === m).transform));
  const targetIds = target.members.filter((m) => !moving.includes(m));
  const tScan = app.scans.get(target.reference) || app.scans.get(targetIds[0]);

  app.assist = {
    sid,
    anchor: sid,
    moving,
    movingSet: new Set(moving),
    wholeGroup,
    ownGroup: own,
    targetGid: target.id,
    targetIds,
    base,
    pose: base.get(sid).clone(),
    offset: new THREE.Matrix4(),
    history: [],
    phase: 'place',
    result: null,
    options: null,
    optIdx: -1,
    colors: 'contrast',
    saved: {
      layout: app.layout,
      hidden: new Set(app.hidden),
      selected: app.selected,
      highlightPair: app.highlightPair,
    },
  };
  app.layout = 'true';
  app.highlightPair = null;
  updateOffsets();
  app.assist.offset = (tScan?.offset || new THREE.Matrix4()).clone();
  assistPlaceBeside();
  app.hidden = new Set(st.scans.filter((s) => !app.assist.movingSet.has(s.id) && !targetIds.includes(s.id)).map((s) => s.id));
  document.body.classList.add('assist-mode');
  buildAssistGizmo();
  updateOffsets();
  fitAll([...moving, ...targetIds]);
  renderAssistPanel();
  requestRender();
}

// Put the moving cloud next to the target (not on top of it), floors level.
function assistPlaceBeside() {
  const A = app.assist;
  const tPts = regSample(A.targetIds, (id) => mat4FromRow(app.scans.get(id).data.transform), 2500);
  const mPts = regSample(A.moving, (id) => A.base.get(id), 2500);
  if (!tPts.length || !mPts.length) return;
  const tx = tPts.map((p) => p.x), ty = tPts.map((p) => p.y), tz = tPts.map((p) => p.z);
  const mx = mPts.map((p) => p.x), my = mPts.map((p) => p.y), mz = mPts.map((p) => p.z);
  const tMaxX = quantile(tx, 0.98), tCy = (quantile(ty, 0.02) + quantile(ty, 0.98)) / 2;
  const mMinX = quantile(mx, 0.02), mCy = (quantile(my, 0.02) + quantile(my, 0.98)) / 2;
  const dz = quantile(tz, 0.03) - quantile(mz, 0.03);
  const shift = new THREE.Matrix4().makeTranslation(tMaxX - mMinX + 2.0, tCy - mCy, dz);
  A.pose = shift.multiply(A.base.get(A.anchor).clone());
}

function stopAssist(restoreView = true) {
  const A = app.assist;
  if (!A) return;
  app.assist = null;
  app.layout = A.saved.layout;
  app.hidden = A.saved.hidden;
  app.highlightPair = A.saved.highlightPair;
  document.body.classList.remove('assist-mode');
  assistGizmo.visible = false;
  $('#assist-panel').classList.add('hidden');
  updateOffsets();
  if (restoreView) fitAll();
  renderScans();
  requestRender();
}

function buildAssistGizmo() {
  assistGizmo.clear();
  const A = app.assist;
  const bb = app.scans.get(A.anchor)?.data.bbox;
  const r = bb ? Math.max(1.5, 0.42 * Math.hypot(bb[1][0] - bb[0][0], bb[1][1] - bb[0][1])) : 5;
  A.ringR = r;
  const pts = [];
  for (let i = 0; i <= 96; i++) {
    const a = (i / 96) * Math.PI * 2;
    pts.push(Math.cos(a) * r, Math.sin(a) * r, 0);
  }
  const mat = (c, o = 1) => new THREE.LineBasicMaterial({ color: c, transparent: true, opacity: o, depthTest: false });
  const ring = new THREE.Line(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(pts, 3)), mat(0xff9f45, 0.9));
  ring.renderOrder = 10;
  assistGizmo.add(ring);
  // heading tick + small arrows so it reads as a rotation handle
  const tick = [r * 0.82, 0, 0, r * 1.12, 0, 0];
  const arrows = [];
  for (const a0 of [Math.PI / 2, -Math.PI / 2, Math.PI]) {
    const c = Math.cos(a0), s = Math.sin(a0);
    const p = [c * r, s * r];
    const t = [-s, c];
    const k = r * 0.07;
    arrows.push(p[0], p[1], 0, p[0] - t[0] * k + c * k * 0.6, p[1] - t[1] * k + s * k * 0.6, 0);
    arrows.push(p[0], p[1], 0, p[0] - t[0] * k - c * k * 0.6, p[1] - t[1] * k - s * k * 0.6, 0);
  }
  const deco = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute([...tick, ...arrows], 3)),
    mat(0xffd2a1, 0.95)
  );
  deco.renderOrder = 10;
  assistGizmo.add(deco);
  const cross = [-r * 0.08, 0, 0, r * 0.08, 0, 0, 0, -r * 0.08, 0, 0, r * 0.08, 0];
  const cx = new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(cross, 3)), mat(0xffffff, 0.8));
  cx.renderOrder = 10;
  assistGizmo.add(cx);
  assistGizmo.visible = true;
}

function updateAssistGizmo() {
  const A = app.assist;
  if (!A) return;
  const c = assistPivot();
  const R = new THREE.Matrix4().extractRotation(A.pose);
  const x = new THREE.Vector3(1, 0, 0).applyMatrix4(R);
  const yaw = Math.atan2(x.y, x.x);
  const M = A.offset.clone()
    .multiply(new THREE.Matrix4().makeTranslation(c.x, c.y, c.z))
    .multiply(new THREE.Matrix4().makeRotationZ(yaw));
  assistGizmo.matrix.copy(M);
  assistGizmo.matrixWorldNeedsUpdate = true;
}

// cheap per-frame update while dragging: only the moving clouds + gizmo
function assistApply() {
  const A = app.assist;
  if (!A) return;
  for (const sid of A.moving) {
    const s = app.scans.get(sid);
    if (!s?.points) continue;
    const M = assistDisplayMatrix(sid);
    for (const obj of [s.points, s.featureEdge, s.featurePlane]) {
      if (!obj) continue;
      obj.matrix.copy(M);
      obj.matrixWorldNeedsUpdate = true;
    }
  }
  updateAssistGizmo();
  requestRender();
}

function assistPush() {
  const A = app.assist;
  A.history.push(A.pose.clone());
  if (A.history.length > 200) A.history.shift();
}

function assistMoved() {
  const A = app.assist;
  if (A.phase === 'result') {
    A.phase = 'place';
    A.result = null;
  }
  assistApply();
  renderAssistPanel();
}

function assistTranslate(dx, dy, dz) {
  const A = app.assist;
  A.pose = new THREE.Matrix4().makeTranslation(dx, dy, dz).multiply(A.pose);
}

function assistRotate(rad) {
  const A = app.assist;
  const c = assistPivot();
  const M = new THREE.Matrix4().makeTranslation(c.x, c.y, c.z)
    .multiply(new THREE.Matrix4().makeRotationZ(rad))
    .multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
  A.pose = M.multiply(A.pose);
}

function assistUndo() {
  const A = app.assist;
  if (!A || !A.history.length) return;
  A.pose = A.history.pop();
  assistMoved();
}

function animateAssistPose(to, ms = 450) {
  const A = app.assist;
  const from = A.pose.clone();
  const p0 = new THREE.Vector3(), q0 = new THREE.Quaternion(), s0 = new THREE.Vector3();
  const p1 = new THREE.Vector3(), q1 = new THREE.Quaternion(), s1 = new THREE.Vector3();
  from.decompose(p0, q0, s0);
  to.decompose(p1, q1, s1);
  const t0 = performance.now();
  return new Promise((resolve) => {
    const step = () => {
      if (app.assist !== A) return resolve();
      const t = Math.min(1, (performance.now() - t0) / ms);
      const e = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      const p = p0.clone().lerp(p1, e);
      const q = q0.clone().slerp(q1, e);
      A.pose = new THREE.Matrix4().compose(p, q, new THREE.Vector3(1, 1, 1));
      assistApply();
      if (t < 1) requestAnimationFrame(step);
      else { A.pose = to.clone(); assistApply(); resolve(); }
    };
    requestAnimationFrame(step);
  });
}

function assistBusy() {
  // manual placement runs in its own lane beside a background registration
  const kind = app.state?.busy_kind;
  if (app.state?.busy && (kind === 'register' || kind === 'improve')) return !!app.assistJob;
  return !!(app.job || app.state?.busy || app.assistJob);
}

// poll an assist job without taking over the main progress bar / Stop button
async function assistJob(resPromise) {
  const res = await resPromise;
  if (!res?.job) return null;
  app.assistJob = res.job;
  try {
    while (true) {
      const j = await getJSON(`/api/jobs/${res.job}`);
      if (app.assist && j.message) { app.assist.busyLabel = j.message; renderAssistPanel(); }
      if (j.status === 'done') return j.result;
      if (j.status === 'error') { toast(j.error || 'failed', 'err'); return null; }
      await sleep(400);
    }
  } finally {
    app.assistJob = null;
  }
}

async function assistSnap() {
  const A = app.assist;
  if (!A || A.phase === 'busy') return;
  if (assistBusy()) {
    toast('Another job is running — wait for it (or press Stop), then Snap.', 'err');
    return;
  }
  A.phase = 'busy';
  A.busyLabel = 'Snapping…';
  renderAssistPanel();
  let res = null;
  try {
    res = await assistJob(
      postJSON('/api/assist/snap', { moving: A.moving, anchor: A.anchor, pose: matRows(A.pose), target_group: A.targetGid })
    );
  } catch (err) {
    toast(errText(err), 'err');
  }
  if (app.assist !== A) return;
  if (!res || res.error) {
    A.phase = 'place';
    renderAssistPanel();
    if (res?.error) toast(res.error, 'err');
    return;
  }
  assistPush();
  await animateAssistPose(mat4FromRow(res.pose));
  A.phase = 'result';
  A.result = res;
  A.snappedPose = A.pose.clone();
  renderAssistPanel();
}

async function assistSuggest() {
  const A = app.assist;
  if (!A || A.phase === 'busy') return;
  if (assistBusy()) {
    toast('Another job is running — press Stop (top right) or wait.', 'err');
    return;
  }
  A.phase = 'busy';
  A.busyLabel = 'Looking for places it fits…';
  renderAssistPanel();
  let res = null;
  try {
    res = await assistJob(
      postJSON('/api/assist/suggest', { moving: A.moving, anchor: A.anchor, target_group: A.targetGid })
    );
  } catch (err) {
    toast(errText(err), 'err');
  }
  if (app.assist !== A) return;
  A.phase = 'place';
  A.result = null;
  A.options = res?.options || [];
  A.optIdx = -1;
  renderAssistPanel();
  if (A.options.length) await assistPickOption(0);
  else toast('No automatic placement found — drag it roughly into place and press Snap.', '');
}

async function assistPickOption(i) {
  const A = app.assist;
  const o = A?.options?.[i];
  if (!o) return;
  A.optIdx = i;
  assistPush();
  A.phase = 'place';
  A.result = null;
  renderAssistPanel();
  await animateAssistPose(mat4FromRow(o.pose));
  renderAssistPanel();
}

async function assistAccept(force = false) {
  const A = app.assist;
  if (!A || !A.result) return;
  try {
    const res = await postJSON('/api/assist/accept', {
      moving: A.moving, anchor: A.anchor, pose: matRows(A.pose), target_group: A.targetGid, force,
      markers: A.markers || null,
    });
    const what = A.moving.length > 1 ? `${scanLabel(A.anchor)} + ${A.moving.length - 1}` : scanLabel(A.anchor);
    const letter = String.fromCharCode(65 + Number(A.targetGid));
    stopAssist(false);
    app.lastHash = '';
    await refreshState(true);
    const g = groupOf(A.anchor);
    focusGroup(g ? g.id : 'all');
    toast(`${what} placed in group ${letter} · ${res.links} link${res.links === 1 ? '' : 's'} saved`, 'ok', {
      label: 'Undo',
      fn: async () => {
        try {
          await postJSON('/api/assist/undo', {});
          app.lastHash = '';
          await refreshState(true);
          focusGroup('all');
          toast('Placement undone', 'ok');
        } catch (err) { toast(errText(err), 'err'); }
      },
    });
  } catch (err) {
    const msg = errText(err);
    if (!force && /accept anyway/i.test(msg)) {
      toast(msg, 'err', { label: 'Accept anyway', fn: () => assistAccept(true) });
    } else {
      toast(msg, 'err');
    }
  }
}

function renderAssistPanel() {
  const A = app.assist;
  const host = $('#assist-panel');
  if (!A) { host.classList.add('hidden'); return; }
  host.classList.remove('hidden');
  const st = app.state;
  const choices = assistTargetChoices(A.moving);
  const letter = (gid) => String.fromCharCode(65 + Number(gid));
  const own = A.ownGroup && !A.ownGroup.unconnected && A.ownGroup.members.length > 1 ? A.ownGroup : null;
  const busy = A.phase === 'busy';
  const res = A.result;
  const step = res ? 3 : 1 + (A.history.length > 0 ? 1 : 0);
  const steps = [
    ['1', 'Drag it roughly into place'],
    ['2', 'Snap'],
    ['3', 'Accept'],
  ].map(([n, t], i) => `<span class="as-step ${i + 1 < step ? 'done' : i + 1 === step ? 'on' : ''}"><b>${i + 1 < step ? '✓' : n}</b>${t}</span>`).join('<i class="as-arrow">›</i>');

  let body = '';
  if (busy) {
    body = `<div class="as-status busy"><span class="spin"></span>${escapeHtml(A.busyLabel || 'Working…')}</div>`;
  } else if (res) {
    const q = res.quality || {};
    const b = res.best || {};
    const moved = b.shift_m != null ? `Snap moved it ${Number(b.shift_m).toFixed(2)} m and turned it ${Number(b.turn_deg || 0).toFixed(1)}°` : '';
    body = `<div class="as-status ${q.level}">
        <span class="as-dot"></span>
        <div><b>${escapeHtml(q.headline || '')}</b><div class="as-detail">${escapeHtml(q.detail || '')}</div>
        <div class="as-detail">matched against ${escapeHtml((res.targets && res.targets.length > 1) ? res.targets.join(', ') : (b.target_label || ''))}${moved ? ' · ' + moved : ''}</div></div>
      </div>
      ${res.snap_note ? `<div class="as-tip">Snap found no better fit near your placement (${escapeHtml(res.snap_note)}), so it stayed exactly where you put it.</div>` : ''}
      ${!b.accepted && b.reason ? `<div class="as-tip">Not verified automatically: ${escapeHtml(b.reason)}. If the walls sit on top of each other in the side and 3D views, accept it anyway — your placement is kept as a fixed link.</div>` : ''}
      ${q.level === 'bad' ? '<div class="as-tip">Tip: Snap only searches within ~60 cm and ~5° of where you drop it — drag it closer to where it belongs, or try <b>Find it for me</b>.</div>' : ''}
      ${q.level === 'ok' && b.accepted ? '<div class="as-tip">Look at the side view and the 3D view: walls should sit on top of each other.</div>' : ''}`;
  } else if (A.history.length === 0 && !A.options?.length) {
    body = `<div class="as-tip big">Drag <b style="color:${ASSIST_MOVING}">${escapeHtml(scanLabel(A.anchor))}</b> in the top view onto the matching spot in <b style="color:${ASSIST_TARGET}">group ${letter(A.targetGid)}</b>. Drag the ring to turn it. Rough is fine — within about a metre.</div>`;
  } else {
    body = `<div class="as-tip">Close enough? Press <b>Snap</b> (or Enter) and it will lock onto the walls.</div>`;
  }

  const opts = A.options?.length
    ? `<div class="as-opts"><span class="muted">Suggestions</span>${A.options.map((o, i) => `
        <button class="as-opt ${i === A.optIdx ? 'on' : ''}" data-as-opt="${i}" title="${escapeHtml(o.method || '')}">
          ${i + 1}<small>${escapeHtml(o.target_label)} · ${Math.round((o.rank || 0) * 100)}%${o.votes > 1 ? ` · ${o.votes}×` : ''}</small>
        </button>`).join('')}</div>`
    : '';

  // The button follows what the server will accept: a verified snap is a
  // plain Accept; anything else needs an explicit "accept anyway".
  const verified = !!res?.best?.accepted;
  const primary = res
    ? (verified
      ? `<button class="btn" data-as="keep">Keep adjusting</button>
         <button class="btn primary as-big" data-as="accept">Accept ✓</button>`
      : res.quality?.level === 'bad'
        ? `<button class="btn small" data-as="accept-force" title="Save it here even though the check failed">Accept anyway</button>
           <button class="btn primary" data-as="keep">Keep adjusting</button>`
        : `<button class="btn" data-as="keep">Keep adjusting</button>
           <button class="btn primary as-big" data-as="accept-force" title="The automatic check did not pass; save it as a fixed link because you checked it">Accept anyway ✓</button>`)
    : `<button class="btn primary as-big" data-as="snap" ${busy || assistBusy() ? 'disabled' : ''} title="${assistBusy() ? 'Another job is running' : 'Lock onto the walls (Enter)'}">Snap ⏎</button>`;

  host.innerHTML = `
    <div class="as-head">
      <div class="as-title">Place <b style="color:${ASSIST_MOVING}">${escapeHtml(scanLabel(A.anchor))}${A.moving.length > 1 ? ` + ${A.moving.length - 1}` : ''}</b> into
        <select data-as="target" ${busy ? 'disabled' : ''}>
          ${choices.map((g) => `<option value="${g.id}" ${g.id === A.targetGid ? 'selected' : ''}>group ${letter(g.id)} · ${g.members.length} scan${g.members.length === 1 ? '' : 's'}${g.id === 0 ? ' (main)' : ''}</option>`).join('')}
        </select>
      </div>
      <button class="as-close" data-as="cancel" title="Cancel (Esc)">×</button>
    </div>
    <div class="as-steps">${steps}</div>
    ${own ? `<label class="check as-whole"><input type="checkbox" data-as="whole" ${A.wholeGroup ? 'checked' : ''} ${busy ? 'disabled' : ''}/> move its whole group ${letter(own.id)} (${own.members.length} scans) together</label>` : ''}
    ${body}
    ${assistManualList(A)}
    ${opts}
    <div class="as-actions">
      <button class="btn small" data-as="suggest" ${busy ? 'disabled' : ''} title="Search the group for places this scan fits">✨ Find it for me</button>
      <button class="btn small" data-as="markers" ${busy ? 'disabled' : ''} title="Click the same 2-5 spots in both panoramas — for scans geometry alone can't place">📍 Match points</button>
      <span class="as-gap"></span>
      <button class="btn small" data-as="rot-l" ${busy ? 'disabled' : ''} title="Turn 90° left ( [ )">↺ 90°</button>
      <button class="btn small" data-as="rot-r" ${busy ? 'disabled' : ''} title="Turn 90° right ( ] )">↻ 90°</button>
      <button class="btn small" data-as="undo" ${busy || !A.history.length ? 'disabled' : ''} title="Undo last move (⌘Z)">Undo</button>
      ${primary}
    </div>
    <div class="as-keys">
      <span><kbd>drag</kbd> move</span><span><kbd>drag ring</kbd> turn</span><span><kbd>space</kbd>+drag / right-drag pan</span>
      <span><kbd>scroll</kbd> zoom</span><span><kbd>Q</kbd><kbd>E</kbd> turn 1° (⇧10°)</span><span><kbd>←↑→↓</kbd> nudge 10 cm (⇧1 m)</span>
      <span><kbd>PgUp</kbd><kbd>PgDn</kbd> height</span><span><kbd>Esc</kbd> cancel</span>
      <span class="as-colors">colours <button class="link ${A.colors === 'contrast' ? 'on' : ''}" data-as="colors-contrast">contrast</button><button class="link ${A.colors === 'rgb' ? 'on' : ''}" data-as="colors-rgb">rgb</button></span>
    </div>`;
}

$('#assist-panel').addEventListener('click', async (e) => {
  const A = app.assist;
  if (!A) return;
  const del = e.target.closest('[data-manual-del]');
  if (del) { await deleteManualLink(...del.dataset.manualDel.split('|')); return; }
  const opt = e.target.closest('[data-as-opt]');
  if (opt) { await assistPickOption(Number(opt.dataset.asOpt)); return; }
  const act = e.target.closest('[data-as]')?.dataset.as;
  if (!act || act === 'target' || act === 'whole') return;
  if (act === 'cancel') stopAssist();
  else if (act === 'snap') await assistSnap();
  else if (act === 'suggest') await assistSuggest();
  else if (act === 'markers') await openPanoPicker();
  else if (act === 'accept') await assistAccept(false);
  else if (act === 'accept-force') await assistAccept(true);
  else if (act === 'keep') { A.phase = 'place'; A.result = null; renderAssistPanel(); }
  else if (act === 'undo') assistUndo();
  else if (act === 'rot-l' || act === 'rot-r') { assistPush(); assistRotate(act === 'rot-l' ? Math.PI / 2 : -Math.PI / 2); assistMoved(); }
  else if (act === 'colors-contrast' || act === 'colors-rgb') { A.colors = act.slice(7); renderAssistPanel(); requestRender(); }
});

$('#assist-panel').addEventListener('change', (e) => {
  const A = app.assist;
  if (!A) return;
  const act = e.target.dataset.as;
  if (act === 'target') startAssist(A.sid, { wholeGroup: A.wholeGroup, targetGid: e.target.value });
  if (act === 'whole') startAssist(A.sid, { wholeGroup: e.target.checked, targetGid: A.targetGid });
});

// screen-space helpers for the top view
function assistScreenOf(paneName, world) {
  const pane = panesCSS.find((p) => p.view === paneName);
  const cam = views[paneName].camera;
  updateOrtho(views[paneName], pane);
  const v = world.clone().project(cam);
  return { x: pane.x + (v.x + 1) / 2 * pane.w, y: pane.y + (1 - v.y) / 2 * pane.h, pane };
}

const assistKeys = { space: false };

// returns true when the assist consumed the event
function assistPointerDown(e) {
  const A = app.assist;
  if (!A || A.phase === 'busy') return false;
  const pane = paneAt(e.clientX, e.clientY);
  if (!pane || pane.view === 'three') return false;
  if (e.button !== 0 || assistKeys.space) return false; // pan as usual
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  let mode = pane.view === 'top' ? 'move' : 'move-side';
  if (pane.view === 'top') {
    const cDisp = assistPivot().applyMatrix4(A.offset);
    const sc = assistScreenOf('top', cDisp);
    const worldPerPx = (2 * views.top.half) / Math.max(sc.pane.h, 1);
    const rPx = A.ringR / worldPerPx;
    const d = Math.hypot(x - sc.x, y - sc.y);
    if (e.shiftKey || Math.abs(d - rPx) < Math.max(10, rPx * 0.12)) {
      mode = 'rotate';
      A.drag = { mode, cx: sc.x, cy: sc.y, a0: Math.atan2(-(y - sc.y), x - sc.x) };
    }
  }
  if (mode !== 'rotate') A.drag = { mode, x, y, pane: pane.view };
  assistPush();
  canvas.setPointerCapture(e.pointerId);
  document.body.classList.add('assist-dragging');
  return true;
}

function assistPointerMove(e) {
  const A = app.assist;
  const r = canvas.getBoundingClientRect();
  const x = e.clientX - r.left, y = e.clientY - r.top;
  if (!A?.drag) {
    // hover feedback: ring = rotate cursor
    if (A && A.phase !== 'busy') {
      const pane = paneAt(e.clientX, e.clientY);
      let cur = '';
      if (assistKeys.space) cur = 'grab';
      else if (pane?.view === 'top') {
        const sc = assistScreenOf('top', assistPivot().applyMatrix4(A.offset));
        const worldPerPx = (2 * views.top.half) / Math.max(sc.pane.h, 1);
        const rPx = A.ringR / worldPerPx;
        cur = Math.abs(Math.hypot(x - sc.x, y - sc.y) - rPx) < Math.max(10, rPx * 0.12) ? 'alias' : 'move';
      } else if (pane?.view === 'side') cur = 'move';
      canvas.style.cursor = cur;
    }
    return false;
  }
  const d = A.drag;
  if (d.mode === 'rotate') {
    const a = Math.atan2(-(y - d.cy), x - d.cx);
    let da = a - d.a0;
    if (e.altKey) da *= 0.1; // fine
    d.a0 = a;
    assistRotate(da);
  } else {
    const view = views[d.pane];
    const pane = panesCSS.find((p) => p.view === d.pane);
    let worldPerPx = (2 * view.half) / Math.max(pane.h, 1);
    if (e.altKey) worldPerPx *= 0.1; // fine
    const dx = (x - d.x) * worldPerPx, dy = (y - d.y) * worldPerPx;
    d.x = x; d.y = y;
    if (d.mode === 'move') assistTranslate(dx, -dy, 0);
    else assistTranslate(dx, 0, -dy);
  }
  assistMoved();
  return true;
}

function assistPointerUp() {
  const A = app.assist;
  if (!A?.drag) return false;
  A.drag = null;
  document.body.classList.remove('assist-dragging');
  renderAssistPanel();
  return true;
}

function assistKeyDown(e) {
  const A = app.assist;
  if (!A) return false;
  const tag = document.activeElement?.tagName;
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(tag)) return false;
  if (e.key === ' ') { assistKeys.space = true; canvas.style.cursor = 'grab'; e.preventDefault(); return true; }
  if (e.key === 'Escape') { stopAssist(); return true; }
  if (A.phase === 'busy') return true;
  if (e.key === 'Enter') {
    if (A.result && A.result.best?.accepted) assistAccept(false);
    else if (A.result && A.result.quality?.level === 'ok') assistAccept(true);
    else assistSnap();
    e.preventDefault();
    return true;
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { assistUndo(); e.preventDefault(); return true; }
  const big = e.shiftKey, fine = e.altKey;
  const stepM = big ? 1.0 : fine ? 0.01 : 0.1;
  const stepA = rotRad(big ? 10 : fine ? 0.1 : 1);
  const k = e.key;
  let did = true;
  assistPush();
  if (k === 'ArrowLeft') assistTranslate(-stepM, 0, 0);
  else if (k === 'ArrowRight') assistTranslate(stepM, 0, 0);
  else if (k === 'ArrowUp') assistTranslate(0, stepM, 0);
  else if (k === 'ArrowDown') assistTranslate(0, -stepM, 0);
  else if (k === 'PageUp') assistTranslate(0, 0, big ? 0.5 : 0.05);
  else if (k === 'PageDown') assistTranslate(0, 0, big ? -0.5 : -0.05);
  else if (k.toLowerCase() === 'q') assistRotate(stepA);
  else if (k.toLowerCase() === 'e') assistRotate(-stepA);
  else if (k === '[') assistRotate(Math.PI / 2);
  else if (k === ']') assistRotate(-Math.PI / 2);
  else { did = false; A.history.pop(); }
  if (did) { e.preventDefault(); assistMoved(); }
  return did;
}

window.addEventListener('keyup', (e) => {
  if (e.key === ' ') { assistKeys.space = false; if (app.assist) canvas.style.cursor = ''; }
});

/* ------------------------------------------------------------------ boot */

// Handle for debugging and browser tests from the console.
window.__e57 = { app, scene, root, views, orbit, fitAll, updateOffsets, get renderer() { return renderer; } };

/* ------------------------------------------------------- panorama markers
 * Two levelled 360° panoramas (target on top, the scan being placed below).
 * Every pixel is a measured point, so clicking the same corner in both gives
 * a 3D pair; 2 pairs fix a levelled scan, 3-5 spread out also check it.
 * The solved pose then goes through the normal Snap (trust region) and the
 * usual Accept, so a marker placement ends up as a regular manual link.
 */
const PANO_COLORS = ['#ffd93d', '#4ecdc4', '#ff6b6b', '#a78bfa', '#6bcb77', '#f97316', '#38bdf8', '#f472b6'];

function panoProject(meta, local) {
  const d = [local[0] - meta.origin[0], local[1] - meta.origin[1], local[2] - meta.origin[2]];
  const r = Math.hypot(d[0], d[1], d[2]);
  if (r < 1e-6) return null;
  const az = Math.atan2(d[1], d[0]);
  const el = Math.asin(Math.max(-1, Math.min(1, d[2] / r)));
  let u = ((Math.PI - az) / (2 * Math.PI)) * meta.w;
  u = ((u % meta.w) + meta.w) % meta.w;
  return { u, v: ((Math.PI / 2 - el) / Math.PI) * meta.h };
}
function mulPoint(rows, p) {
  return [0, 1, 2].map((i) => rows[i][0] * p[0] + rows[i][1] * p[1] + rows[i][2] * p[2] + rows[i][3]);
}
function invRows(rows) {
  const m = mat4FromRow(rows).invert();
  return matRows(m);
}

async function openPanoPicker() {
  const A = app.assist;
  if (!A || A.phase === 'busy') return;
  // default target: the scan of the target group nearest to where the moving scan sits now
  const here = new THREE.Vector3().setFromMatrixPosition(A.pose);
  const st = app.state;
  const targets = A.targetIds.filter((id) => st.scans.find((s) => s.id === id)?.loaded);
  if (!targets.length) { toast('No loaded scan to match against', 'err'); return; }
  const dist = (id) => new THREE.Vector3().setFromMatrixPosition(mat4FromRow(st.scans.find((s) => s.id === id).transform)).distanceTo(here);
  targets.sort((a, b) => dist(a) - dist(b));
  app.pano = { moving: A.anchor, target: targets[0], targets, pairs: [], solved: null, meta: {}, img: {}, view: {}, active: 't' };
  $('#pano').classList.remove('hidden');
  renderPano();
  await panoLoad();
}

function closePano() {
  app.pano = null;
  $('#pano').classList.add('hidden');
}

async function panoLoad() {
  const P = app.pano;
  if (!P) return;
  for (const side of ['t', 'm']) $(`.pano-view[data-side="${side}"] .pano-loading`).classList.remove('hidden');
  const res = await assistJob(postJSON('/api/pano/prepare', { ids: [P.target, P.moving] }));
  if (app.pano !== P) return;
  if (!res?.panoramas) { toast('Could not build the panoramas', 'err'); closePano(); return; }
  P.meta = res.panoramas;
  await Promise.all(['t', 'm'].map((side) => new Promise((resolve) => {
    const sid = side === 't' ? P.target : P.moving;
    const img = new Image();
    img.onload = () => { P.img[side] = img; resolve(); };
    img.onerror = () => resolve();
    img.src = `/api/scans/${encodeURIComponent(sid)}/pano.jpg`;
  })));
  if (app.pano !== P) return;
  for (const side of ['t', 'm']) {
    $(`.pano-view[data-side="${side}"] .pano-loading`).classList.add('hidden');
    panoFit(side);
  }
  await panoRestore(P);
  renderPano();
}

// markers are saved on every change and restored when the picker reopens
async function panoRestore(P) {
  let saved = null;
  try { saved = (await postJSON('/api/pano/markers/load', { moving: P.moving, target: P.target })).markers; } catch { return; }
  if (app.pano !== P || !saved?.pairs?.length) return;
  const pick = async (sid, uv) => {
    if (!uv) return null;
    try {
      const r = await postJSON('/api/pano/pick', { sid, u: uv[0], v: uv[1] });
      return r.ok ? { u: r.u, v: r.v, local: r.local, world: r.world } : null;
    } catch { return null; }
  };
  const pairs = [];
  for (const sp of saved.pairs) {
    const pr = {};
    const t = await pick(P.target, sp.t), m = await pick(P.moving, sp.m);
    if (t) pr.t = t;
    if (m) pr.m = m;
    if (pr.t || pr.m) pairs.push(pr);
  }
  if (app.pano !== P) return;
  P.pairs = pairs;
  P.history = pairs.flatMap((pr) => ['t', 'm'].filter((k) => pr[k]).map((key) => ({ pair: pr, key })));
  P.restored = new Date(saved.created * 1000);
  P.restoring = true;
  try { await panoSolve(); } finally { P.restoring = false; }
  toast(`Restored ${pairs.length} saved marker pair${pairs.length === 1 ? '' : 's'}`, 'ok');
}

function panoSave() {
  const P = app.pano;
  if (!P) return;
  const pairs = P.pairs.map((p) => ({ t: p.t ? [p.t.u, p.t.v] : null, m: p.m ? [p.m.u, p.m.v] : null,
    t_world: p.t?.world || null, m_local: p.m?.local || null }));
  postJSON('/api/pano/markers/save', { moving: P.moving, target: P.target, pairs }).catch(() => {});
}

function panoCanvas(side) { return $(`.pano-view[data-side="${side}"] canvas`); }

function panoFit(side) {
  const P = app.pano, c = panoCanvas(side), m = P.meta[side === 't' ? P.target : P.moving];
  if (!m) return;
  const w = c.clientWidth, h = c.clientHeight;
  const scale = Math.max(w / m.w, h / (m.h * 0.75));  // full width, most of the height
  P.view[side] = { scale, x: 0, y: -(m.h * 0.08) * scale };
}

// panorama pixel <-> canvas pixel (CSS px); u wraps around
function panoToScreen(side, u, v) {
  const V = app.pano.view[side], m = app.pano.meta[side === 't' ? app.pano.target : app.pano.moving];
  const W = m.w * V.scale;
  let x = V.x + u * V.scale;
  x = ((x % W) + W) % W;
  return { x, y: V.y + v * V.scale, W };
}
function panoFromScreen(side, x, y) {
  const V = app.pano.view[side], m = app.pano.meta[side === 't' ? app.pano.target : app.pano.moving];
  let u = (x - V.x) / V.scale;
  u = ((u % m.w) + m.w) % m.w;
  return { u, v: (y - V.y) / V.scale };
}

function panoCenterOn(side, u, v) {
  const c = panoCanvas(side), V = app.pano.view[side];
  V.x = c.clientWidth / 2 - u * V.scale;
  V.y = c.clientHeight / 2 - v * V.scale;
}

function drawPano(side) {
  const P = app.pano, c = panoCanvas(side);
  if (!P || !c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);
  const img = P.img[side], V = P.view[side];
  if (!img || !V) return;
  const W = img.width * V.scale, H = img.height * V.scale;
  let x0 = ((V.x % W) + W) % W - W;
  ctx.imageSmoothingEnabled = V.scale < 2;
  for (let x = x0; x < w; x += W) ctx.drawImage(img, x, V.y, W, H);
  // markers
  const key = side === 't' ? 't' : 'm';
  P.pairs.forEach((pr, i) => {
    const mk = pr[key];
    if (!mk) return;
    const col = PANO_COLORS[i % PANO_COLORS.length];
    for (const dx of [0, -W, W]) {
      const p = panoToScreen(side, mk.u, mk.v);
      const x = p.x + dx, y = p.y;
      if (x < -20 || x > w + 20) continue;
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#000';
      ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = col;
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(x - 13, y); ctx.lineTo(x - 4, y); ctx.moveTo(x + 4, y); ctx.lineTo(x + 13, y);
      ctx.moveTo(x, y - 13); ctx.lineTo(x, y - 4); ctx.moveTo(x, y + 4); ctx.lineTo(x, y + 13); ctx.stroke();
      ctx.fillStyle = col;
      ctx.font = 'bold 12px -apple-system, sans-serif';
      ctx.fillText(String(i + 1), x + 11, y - 9);
    }
  });
  // where the open pair's partner should be, once the pose is known
  const pred = panoPrediction(side);
  if (pred) {
    const p = panoToScreen(side, pred.u, pred.v);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(p.x, p.y, 16, 0, Math.PI * 2); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#fff';
    ctx.font = '12px -apple-system, sans-serif';
    ctx.fillText('expected here', p.x + 20, p.y + 4);
  }
}

function panoOpenPair() {
  const P = app.pano;
  const last = P.pairs[P.pairs.length - 1];
  return last && !(last.t && last.m) ? last : null;
}

function panoPrediction(side) {
  const P = app.pano, open = panoOpenPair();
  if (!P.solved?.pose || !open) return null;
  if (side === 'm' && open.t && !open.m) {
    const local = mulPoint(invRows(P.solved.pose), open.t.world);
    return panoProject(P.meta[P.moving], local);
  }
  if (side === 't' && open.m && !open.t) {
    const world = mulPoint(P.solved.pose, open.m.local);
    const tRows = app.state.scans.find((s) => s.id === P.target).transform;
    return panoProject(P.meta[P.target], mulPoint(invRows(tRows), world));
  }
  return null;
}

function renderPano() {
  const P = app.pano;
  if (!P) return;
  const lab = (sid) => escapeHtml(scanLabel(sid));
  $('#pano-sub').innerHTML = `placing <b style="color:${ASSIST_MOVING}">${lab(P.moving)}</b> against
    <select id="pano-target">${P.targets.map((t) => `<option value="${t}" ${t === P.target ? 'selected' : ''}>${lab(t)}</option>`).join('')}</select>`;
  $('.pano-view[data-side="t"] .pano-label').innerHTML = `target · ${lab(P.target)}`;
  $('.pano-view[data-side="m"] .pano-label').innerHTML = `<span style="color:${ASSIST_MOVING}">placing · ${lab(P.moving)}</span>`;
  const complete = P.pairs.filter((p) => p.t && p.m).length;
  const open = panoOpenPair();
  let help;
  if (!complete && !open) help = 'Click a sharp, recognisable spot in the <b>top</b> picture — a door or window corner, a light switch, a sign.';
  else if (open && open.t && !open.m) help = `Now click <b>the same spot</b> in the <b>bottom</b> picture${P.solved ? ' — it should be near the dashed circle' : ''}.`;
  else if (open && open.m && !open.t) help = `Now click <b>the same spot</b> in the <b>top</b> picture${P.solved ? ' — it should be near the dashed circle' : ''}.`;
  else if (complete < 2) help = 'One more pair, please — spread out, ideally on another wall.';
  else help = `<b>${complete} pairs.</b> Add 1–3 more spread around the room to double-check, or press <b>Use these points</b>.`;
  $('#pano-help').innerHTML = help;
  for (const side of ['t', 'm']) $(`.pano-view[data-side="${side}"]`).classList.toggle('active',
    !!open && ((side === 'm' && open.t && !open.m) || (side === 't' && open.m && !open.t)));

  const res = P.solved?.residuals || [];
  const used = P.solved?.used || [];
  const rows = P.pairs.map((pr, i) => {
    const k = used.indexOf(i);
    const cm = k >= 0 && res[k] != null ? res[k] * 100 : null;
    const cls = cm == null ? '' : cm < 5 ? 'good' : cm < 15 ? 'ok' : 'bad';
    const state = !(pr.t && pr.m) ? '<span class="muted">waiting for its partner…</span>' : cm == null ? '<span class="muted">—</span>' : '';
    return `<div class="pano-pair"><span class="dot" style="background:${PANO_COLORS[i % PANO_COLORS.length]}">${i + 1}</span>
      ${state}<span class="res ${cls}">${cm == null ? '' : cm.toFixed(1) + ' cm'}</span>
      <button data-pano-del="${i}" title="Remove this pair">×</button></div>`;
  }).join('');
  const worst = res.length ? Math.max(...res) * 100 : null;
  let summary = '';
  if (P.solved?.pose) {
    summary = complete === 2
      ? 'Two pairs fix the position; a third shows whether a click was off.'
      : worst > 15
        ? `<b>One marker looks off</b> (${worst.toFixed(0)} cm) — remove the red one or re-click it more precisely.`
        : `All markers agree within <b>${worst.toFixed(1)} cm</b>. Snap will fine-tune to millimetres.`;
  } else if (P.solved?.error) summary = escapeHtml(P.solved.error);
  $('#pano-list').innerHTML = `<h4>Marker pairs</h4>${rows || '<div class="muted">none yet</div>'}<div class="pano-summary">${summary}</div>
    <div class="pano-summary">Scroll to zoom, drag to pan. Clicks snap to the nearest measured surface.</div>`;
  $('#pano-use').disabled = !(P.solved?.pose && complete >= 2);
  $('#pano-undo').disabled = !P.pairs.length;
  drawPano('t');
  drawPano('m');
}

async function panoSolve() {
  const P = app.pano;
  if (P && !P.restoring) panoSave();
  const pairs = P.pairs.filter((p) => p.t && p.m);
  const idx = P.pairs.map((p, i) => (p.t && p.m ? i : -1)).filter((i) => i >= 0);
  if (pairs.length < 2) { P.solved = null; renderPano(); return; }
  try {
    const r = await postJSON('/api/pano/solve', { moving: P.moving, target: P.target,
      pairs: pairs.map((p) => ({ m: [p.m.u, p.m.v], t: [p.t.u, p.t.v] })) });
    if (app.pano !== P) return;
    // map the server's "used" (index into the complete pairs) back to P.pairs
    if (r.used) r.used = r.used.map((k) => idx[k]);
    P.solved = r;
  } catch (err) {
    P.solved = { error: errText(err) };
  }
  renderPano();
}

async function panoClick(side, x, y) {
  const P = app.pano;
  const m = P.meta[side === 't' ? P.target : P.moving];
  if (!m || !P.view[side]) return;
  const { u, v } = panoFromScreen(side, x, y);
  if (v < 0 || v >= m.h) return;
  const key = side === 't' ? 't' : 'm';
  // an open pair takes the other half; clicking the same picture again
  // moves that marker instead of starting a half-empty pair
  let pr = panoOpenPair();
  if (!pr) { pr = {}; P.pairs.push(pr); }
  const sid = side === 't' ? P.target : P.moving;
  let r;
  try {
    r = await postJSON('/api/pano/pick', { sid, u, v });
  } catch (err) { toast(errText(err), 'err'); return; }
  if (app.pano !== P) return;
  if (!r.ok) {
    if (!pr.t && !pr.m) P.pairs.pop();
    toast(r.reason || 'Nothing measured there — pick a solid surface', 'err');
    renderPano();
    return;
  }
  pr[key] = { u: r.u, v: r.v, local: r.local, world: r.world };
  P.history = (P.history || []).filter((h) => !(h.pair === pr && h.key === key));
  P.history.push({ pair: pr, key });
  // after the first half of a pair, bring the predicted partner into view
  const other = side === 't' ? 'm' : 't';
  const pred = panoPrediction(other);
  if (pred) panoCenterOn(other, pred.u, pred.v);
  if (pr.t && pr.m) await panoSolve();
  else { panoSave(); renderPano(); }
}

async function panoUse() {
  const P = app.pano, A = app.assist;
  if (!P?.solved?.pose || !A) return;
  const pose = mat4FromRow(P.solved.pose);
  // kept with the placement in the permanent manual store
  A.markers = P.pairs.filter((p) => p.t && p.m).map((p, i) => ({
    target: P.target, moving: P.moving,
    t: [p.t.u, p.t.v], m: [p.m.u, p.m.v], t_world: p.t.world, m_local: p.m.local,
    residual_m: P.solved.residuals?.[(P.solved.used || []).indexOf(i)] ?? null,
  }));
  closePano();
  assistPush();
  await animateAssistPose(pose);
  toast('Placed from your markers — snapping to the walls…', 'ok');
  await assistSnap();
}

// pointer handling: drag pans, a click (no drag) places a marker
(function wirePano() {
  for (const side of ['t', 'm']) {
    const view = $(`.pano-view[data-side="${side}"]`);
    const c = view.querySelector('canvas');
    let drag = null;
    c.addEventListener('pointerdown', (e) => {
      if (!app.pano?.view[side]) return;
      c.setPointerCapture(e.pointerId);
      drag = { x: e.clientX, y: e.clientY, vx: app.pano.view[side].x, vy: app.pano.view[side].y, moved: false };
    });
    c.addEventListener('pointermove', (e) => {
      if (!drag || !app.pano) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; view.classList.add('panning'); }
      if (drag.moved) {
        const V = app.pano.view[side];
        V.x = drag.vx + dx; V.y = drag.vy + dy;
        drawPano(side);
      }
    });
    c.addEventListener('pointerup', async (e) => {
      if (!drag) return;
      const d = drag;
      drag = null;
      view.classList.remove('panning');
      if (!d.moved) {
        const rect = c.getBoundingClientRect();
        await panoClick(side, e.clientX - rect.left, e.clientY - rect.top);
      }
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      const V = app.pano?.view[side];
      if (!V) return;
      const rect = c.getBoundingClientRect();
      const x = e.clientX - rect.left, y = e.clientY - rect.top;
      const f = Math.exp(-e.deltaY * 0.0015);
      const ns = Math.min(8, Math.max(0.15, V.scale * f));
      V.x = x - ((x - V.x) * ns) / V.scale;
      V.y = y - ((y - V.y) * ns) / V.scale;
      V.scale = ns;
      drawPano(side);
    }, { passive: false });
  }
  $('#pano-cancel').onclick = closePano;
  $('#pano-use').onclick = panoUse;
  $('#pano-undo').onclick = () => {
    const P = app.pano;
    const h = P?.history?.pop();
    if (!h) return;
    delete h.pair[h.key];
    P.pairs = P.pairs.filter((p) => p.t || p.m);
    panoSolve();
  };
  $('#pano-list').addEventListener('click', (e) => {
    const b = e.target.closest('[data-pano-del]');
    if (!b || !app.pano) return;
    const [gone] = app.pano.pairs.splice(Number(b.dataset.panoDel), 1);
    app.pano.history = (app.pano.history || []).filter((h) => h.pair !== gone);
    panoSolve();
  });
  $('#pano').addEventListener('change', (e) => {
    if (e.target.id === 'pano-target' && app.pano) {
      app.pano.target = e.target.value;
      app.pano.pairs = [];
      app.pano.solved = null;
      app.pano.img = {};
      panoLoad();
    }
  });
  window.addEventListener('keydown', (e) => {
    if (!app.pano) return;
    if (e.key === 'Escape') { e.preventDefault(); e.stopImmediatePropagation(); closePano(); }
    else if (e.key === 'Enter' && !$('#pano-use').disabled) { e.preventDefault(); e.stopImmediatePropagation(); panoUse(); }
    else if (e.key === 'Backspace' && document.activeElement?.tagName !== 'SELECT') { e.preventDefault(); e.stopImmediatePropagation(); $('#pano-undo').click(); }
    else e.stopImmediatePropagation();
  }, true);
  window.addEventListener('resize', () => { if (app.pano) renderPano(); });
})();

/* manual placements (permanent store) shown on scans and in the panel */
function manualLinksOf(sid) {
  return app.state?.manual?.[sid] || [];
}
function manualKindLabel(k) {
  return { markers: 'matched points', snap: 'snapped', force: 'accepted anyway', 'accepted-link': 'accepted link', imported: 'placed earlier', assistant: 'placed by assistant' }[k] || k;
}
function manualPin(sid) {
  const ls = manualLinksOf(sid);
  if (!ls.length) return '';
  const tip = 'Manual links (always win):\n' + ls.map((l) => `· ${l.partner_label} — ${manualKindLabel(l.kind)}${l.markers ? `, ${l.markers} points` : ''}`).join('\n');
  return `<span class="manual-pin" title="${escapeHtml(tip)}">📌${ls.length > 1 ? ls.length : ''}</span>`;
}
async function deleteManualLink(a, b) {
  const A = app.assist;
  if (!confirm(`Delete the manual link ${scanLabel(a)} ↔ ${scanLabel(b)}?\n\nIt is kept in the history but no longer used; the scans are re-solved without it.`)) return;
  try {
    await postJSON('/api/manual/delete', { a, b });
    app.lastHash = '';
    await refreshState(true);
    toast(`Manual link ${scanLabel(a)} ↔ ${scanLabel(b)} deleted`, 'ok');
    // positions may have changed: reopen the placement on fresh poses
    if (A && app.assist === A) startAssist(A.sid, { wholeGroup: A.wholeGroup, targetGid: A.targetGid });
  } catch (err) {
    toast(errText(err), 'err');
  }
}

function assistManualList(A) {
  const rows = [];
  for (const sid of A.moving) {
    for (const l of manualLinksOf(sid)) {
      if (A.movingSet.has(l.partner)) continue;
      const off = l.status === 'contradicted';
      rows.push(`<li class="${off ? 'off' : ''}"><b>${escapeHtml(scanLabel(sid))}</b> ↔ ${escapeHtml(l.partner_label)}
        <span class="muted">· ${escapeHtml(manualKindLabel(l.kind))}${l.markers ? ` · ${l.markers} points` : ''} · ${new Date(l.created * 1000).toLocaleString([], { dateStyle: 'short', timeStyle: 'short' })}</span>
        ${off ? `<span class="as-manual-off" title="${escapeHtml(l.reason || '')}">not used: contradicts newer placements</span>` : ''}
        <button class="as-manual-del" data-manual-del="${escapeHtml(sid)}|${escapeHtml(l.partner)}" title="Delete this manual link (kept in the history, no longer used)">🗑</button></li>`);
    }
  }
  if (!rows.length) return '';
  return `<div class="as-manual"><div class="as-manual-head">📌 Already placed by hand (these always win)</div><ul>${rows.join('')}</ul></div>`;
}

async function boot() {
  layoutViewports();
  await refreshState(true);
  applySavedLayout(app.state || {});
  fitAll();
  const st = app.state;
  if (st && st.scans.length && st.scans.some((s) => !s.loaded) && !st.busy) {
    await runJob(postJSON('/api/load', {}), 'Loading scans');
    app.fitPending = true;
    fitAll();
  }
}

boot().catch((err) => toast(String(err), 'err'));
