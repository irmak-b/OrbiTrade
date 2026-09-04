gsap.registerPlugin(ScrollTrigger);

// 1. Scene, Camera & Renderer
const canvas = document.querySelector('#webgl-canvas');
const scene = new THREE.Scene();

const camera = new THREE.PerspectiveCamera(
  60,
  window.innerWidth / window.innerHeight,
  0.1,
  1000
);
camera.position.z = 25;

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

// 2. Parçacık Sayısı ve Geometri Hazırlığı
const PARTICLE_COUNT = 18000;
const geometry = new THREE.BufferGeometry();

const posStateSpace = new Float32Array(PARTICLE_COUNT * 3);
const posStateChart = new Float32Array(PARTICLE_COUNT * 3);
const colors = new Float32Array(PARTICLE_COUNT * 3);

const colorCyan = new THREE.Color(0x00f0ff);
const colorPurple = new THREE.Color(0x9d4edd);
const colorWhite = new THREE.Color(0xffffff);

// --- SCENARIO 1: Diffuse Star Cluster ---
for (let i = 0; i < PARTICLE_COUNT; i++) {
  const i3 = i * 3;
  posStateSpace[i3] = (Math.random() - 0.5) * 60;
  posStateSpace[i3 + 1] = (Math.random() - 0.5) * 50;
  posStateSpace[i3 + 2] = (Math.random() - 0.5) * 40;

  const mixedColor = Math.random() > 0.4 ? colorCyan : (Math.random() > 0.5 ? colorPurple : colorWhite);
  colors[i3] = mixedColor.r;
  colors[i3 + 1] = mixedColor.g;
  colors[i3 + 2] = mixedColor.b;
}

// --- SCENARIO 2: Screen-Filling Expanded Trade/Candle Chart ---
const CANDLE_COUNT = 30;
const particlesPerCandle = Math.floor(PARTICLE_COUNT / CANDLE_COUNT);

for (let c = 0; c < CANDLE_COUNT; c++) {
  const normX = c / (CANDLE_COUNT - 1); // 0 -> 1
  
  // By setting the multiplier to 46, the two ends of the graph were brought closer to—or extended toward—the screen boundaries.
  const candleX = (normX - 0.5) * 46;
  
  const baseY = Math.pow(normX, 1.3) * 15 - 7 + Math.sin(c * 0.7) * 1.8;
  const candleHeight = 1.8 + Math.random() * 4.0;
  const candleWidth = 0.9;
  const wickHeight = candleHeight + 3.0 + Math.random() * 2.5;

  const startIdx = c * particlesPerCandle;
  const endIdx = (c === CANDLE_COUNT - 1) ? PARTICLE_COUNT : startIdx + particlesPerCandle;

  for (let i = startIdx; i < endIdx; i++) {
    const i3 = i * 3;
    const isWick = Math.random() < 0.22;

    if (isWick) {
      posStateChart[i3] = candleX + (Math.random() - 0.5) * 0.1;
      posStateChart[i3 + 1] = baseY + (Math.random() - 0.5) * wickHeight;
      posStateChart[i3 + 2] = (Math.random() - 0.5) * 0.25;
    } else {
      posStateChart[i3] = candleX + (Math.random() - 0.5) * candleWidth;
      posStateChart[i3 + 1] = baseY + (Math.random() - 0.5) * candleHeight;
      posStateChart[i3 + 2] = (Math.random() - 0.5) * candleWidth;
    }
  }
}

// GPU Attribute Binding
geometry.setAttribute('position', new THREE.BufferAttribute(posStateSpace, 3));
geometry.setAttribute('aPosTarget', new THREE.BufferAttribute(posStateChart, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

// 3. Custom Shader Material
const particleMaterial = new THREE.ShaderMaterial({
  transparent: true,
  depthWrite: false,
  blending: THREE.AdditiveBlending,
  vertexColors: true,
  uniforms: {
    uProgress: { value: 0.0 },
    uTime: { value: 0.0 },
    uMouse: { value: new THREE.Vector2(0, 0) }
  },
  vertexShader: `
    uniform float uProgress;
    uniform float uTime;
    uniform vec2 uMouse;

    attribute vec3 aPosTarget;
    varying vec3 vColor;

    void main() {
      vColor = color;

      vec3 currentPos = mix(position, aPosTarget, uProgress);

      // Mouse Parallax & Pull Force
      float distToMouse = distance(currentPos.xy, uMouse * 18.0);
      float mouseInfluence = smoothstep(7.0, 0.0, distToMouse) * (1.0 - uProgress * 0.6);
      currentPos.xy += normalize(currentPos.xy - (uMouse * 18.0) + 0.001) * mouseInfluence * 1.8;

      // Natural Vibration
      currentPos.y += sin(uTime * 1.5 + currentPos.x * 0.4) * 0.15;
      currentPos.x += cos(uTime * 1.2 + currentPos.y * 0.4) * 0.15;

      vec4 mvPosition = modelViewMatrix * vec4(currentPos, 1.0);
      gl_PointSize = (48.0 / -mvPosition.z) * (1.0 + mouseInfluence * 0.5);
      gl_Position = projectionMatrix * mvPosition;
    }
  `,
  fragmentShader: `
    varying vec3 vColor;

    void main() {
      float dist = length(gl_PointCoord - vec2(0.5));
      if (dist > 0.5) discard;

      float alpha = smoothstep(0.5, 0.05, dist);
      gl_FragColor = vec4(vColor, alpha * 0.9);
    }
  `
});

const points = new THREE.Points(geometry, particleMaterial);
scene.add(points);

// 4. Mouse Interaction
const targetMouse = new THREE.Vector2(0, 0);
window.addEventListener('mousemove', (e) => {
  targetMouse.x = (e.clientX / window.innerWidth) * 2 - 1;
  targetMouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
});

// 5. GSAP ScrollTrigger
gsap.to(particleMaterial.uniforms.uProgress, {
  value: 1.0,
  ease: "power2.inOut",
  scrollTrigger: {
    trigger: ".chart-section",
    start: "top bottom",
    end: "center center",
    scrub: 1.2
  }
});

// Camera Movement
gsap.to(camera.position, {
  z: 23,
  y: 1.5,
  scrollTrigger: {
    trigger: ".chart-section",
    start: "top bottom",
    end: "bottom center",
    scrub: 1.5
  }
});

// 6. Animation Loop
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const elapsedTime = clock.getElapsedTime();

  particleMaterial.uniforms.uTime.value = elapsedTime;
  particleMaterial.uniforms.uMouse.value.lerp(targetMouse, 0.05);

  points.rotation.y = targetMouse.x * 0.06;
  points.rotation.x = -targetMouse.y * 0.06;

  renderer.render(scene, camera);
}
animate();

// 7. Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
});