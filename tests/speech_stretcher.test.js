// Run with: node tests/speech_stretcher.test.js (exits non-zero on failure)
const assert = require('assert');
const path = require('path');
const SpeechStretcher = require(path.join(__dirname, '..', 'static', 'speech_stretcher.js'));

const sr = 24000;

function voicedSignal(seconds, f0) {
    const n = Math.round(seconds * sr);
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const t = i / sr;
        const env = 0.6 + 0.4 * Math.sin(2 * Math.PI * 3 * t);
        out[i] = env * (0.5 * Math.sin(2 * Math.PI * f0 * t)
            + 0.25 * Math.sin(2 * Math.PI * 2 * f0 * t)
            + 0.12 * Math.sin(2 * Math.PI * 3 * f0 * t));
    }
    return out;
}

function estimatePitch(x) {
    const start = Math.floor(x.length / 4);
    const frame = 2048;
    const scores = [];
    let best = 0;
    const minLag = Math.floor(sr / 400);
    for (let lag = minLag; lag <= Math.floor(sr / 70); lag++) {
        let c = 0;
        for (let i = 0; i < frame; i++) c += x[start + i] * x[start + i + lag];
        scores.push(c);
        best = Math.max(best, c);
    }
    // Shortest lag that is nearly as good as the best avoids octave errors.
    const index = scores.findIndex((c, i) => c >= 0.9 * best
        && (i === 0 || c >= scores[i - 1])
        && (i === scores.length - 1 || c >= scores[i + 1]));
    return sr / (minLag + index);
}

function stretch(input, speed, chunk) {
    const s = new SpeechStretcher(sr);
    for (let i = 0; i < input.length; i += chunk) s.push(input.slice(i, i + chunk));
    s.end();
    const parts = [];
    let total = 0;
    while (!s.finished) {
        const o = s.read(2048, speed);
        if (!o.length) break;
        parts.push(o);
        total += o.length;
    }
    const out = new Float32Array(total);
    let p = 0;
    for (const o of parts) { out.set(o, p); p += o.length; }
    return out;
}

const input = voicedSignal(4, 180);

for (const speed of [0.8, 1.2, 1.5, 2]) {
    const out = stretch(input, speed, 1920);
    const effective = input.length / out.length;
    assert.ok(Math.abs(effective - speed) / speed < 0.03,
        `speed ${speed}: effective ${effective.toFixed(3)}`);
    const pitch = estimatePitch(out);
    assert.ok(Math.abs(pitch - 180) / 180 < 0.05,
        `speed ${speed}: pitch ${pitch.toFixed(1)} Hz should stay near 180 Hz`);
}

const normal = stretch(input, 1, 1920);
const odd = stretch(input, 1, 777);
assert.strictEqual(normal.length, odd.length, 'output must not depend on chunk size');
let maxDiff = 0;
for (let i = 0; i < Math.min(normal.length, input.length); i++) {
    maxDiff = Math.max(maxDiff, Math.abs(normal[i] - input[i]));
}
assert.ok(maxDiff < 1e-4, `1x must reproduce the input (max diff ${maxDiff})`);

const silent = stretch(new Float32Array(0), 1.5, 1920);
assert.strictEqual(silent.length, 0, 'empty input gives empty output');

console.log('speech_stretcher: all checks passed');
