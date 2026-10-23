/*
 * Streaming pitch-preserving time stretch (WSOLA) for mono speech.
 * Changes playback speed without changing the voice pitch.
 *
 *   const s = new SpeechStretcher(24000);
 *   s.push(float32Chunk);            // as audio arrives
 *   s.end();                         // when no more input is coming
 *   const out = s.read(2048, 1.2);   // 2048 samples of 1.2x speech
 */
(function (root) {
    'use strict';

    class SpeechStretcher {
        constructor(sampleRate) {
            this.sampleRate = sampleRate;
            this.N = Math.max(256, Math.round((0.04 * sampleRate) / 2) * 2);
            this.H = this.N / 2;
            this.tol = Math.round(0.01 * sampleRate);
            this.window = new Float32Array(this.N);
            for (let i = 0; i < this.N; i++) {
                this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.N);
            }

            this.pending = [];
            this.work = new Float32Array(0);
            this.workBase = 0;
            this.inputEnd = 0;
            this.ended = false;

            this.acc = new Float32Array(this.N);
            this.fifo = new Float32Array(0);
            this.pos = 0;
            this.prevChosen = null;
            this.done = false;
        }

        push(chunk) {
            if (!chunk.length) return;
            this.pending.push(chunk);
            this.inputEnd += chunk.length;
        }

        end() {
            this.ended = true;
        }

        get finished() {
            return this.done && this.fifo.length === 0;
        }

        _available() {
            return this.workBase + this.work.length;
        }

        _ensure(upTo) {
            while (this._available() < upTo && this.pending.length) {
                const next = this.pending.shift();
                const merged = new Float32Array(this.work.length + next.length);
                merged.set(this.work, 0);
                merged.set(next, this.work.length);
                this.work = merged;
            }
        }

        _trim(keepFrom) {
            const drop = Math.min(keepFrom - this.workBase, this.work.length);
            if (drop > 4096) {
                this.work = this.work.slice(drop);
                this.workBase += drop;
            }
        }

        _sample(index) {
            const i = index - this.workBase;
            return i >= 0 && i < this.work.length ? this.work[i] : 0;
        }

        _bestStart(nominal) {
            if (this.prevChosen === null) return Math.max(0, nominal);
            const tpl = this.prevChosen + this.H;
            const len = this.N >> 1;
            let best = Math.max(0, nominal);
            let bestScore = -Infinity;
            for (let k = -this.tol; k <= this.tol; k++) {
                const s = nominal + k;
                if (s < this.workBase) continue;
                let dot = 0;
                let energy = 1e-9;
                for (let i = 0; i < len; i += 2) {
                    const a = this._sample(tpl + i);
                    const b = this._sample(s + i);
                    dot += a * b;
                    energy += b * b;
                }
                const score = dot / Math.sqrt(energy);
                if (score > bestScore) {
                    bestScore = score;
                    best = s;
                }
            }
            return best;
        }

        _makeFrame(speed) {
            if (this.done) return false;
            const nominal = Math.round(this.pos);
            const need = nominal + this.tol + this.N + this.H + 2;
            this._ensure(need);
            if (this._available() < need && !this.ended) return false;
            if (nominal >= this.inputEnd && this.ended) {
                this.done = true;
                if (this.prevChosen !== null) {
                    this._appendFifo(this.acc.slice(0, this.H));
                }
                return false;
            }

            const chosen = this._bestStart(nominal);
            const first = this.prevChosen === null;
            for (let i = 0; i < this.N; i++) {
                const w = first && i < this.H ? 1 : this.window[i];
                this.acc[i] += w * this._sample(chosen + i);
            }
            this._appendFifo(this.acc.slice(0, this.H));
            this.acc.copyWithin(0, this.H);
            this.acc.fill(0, this.H);

            this.prevChosen = chosen;
            this.pos += this.H * speed;
            this._trim(Math.min(chosen, nominal) - this.tol - 1);
            return true;
        }

        _appendFifo(samples) {
            const merged = new Float32Array(this.fifo.length + samples.length);
            merged.set(this.fifo, 0);
            merged.set(samples, this.fifo.length);
            this.fifo = merged;
        }

        /** Return up to `count` samples at `speed` (1 = normal). May return fewer, or none, if input is short. */
        read(count, speed) {
            const s = Math.min(4, Math.max(0.25, speed || 1));
            while (this.fifo.length < count && this._makeFrame(s)) { /* keep producing */ }
            const n = Math.min(count, this.fifo.length);
            const out = this.fifo.slice(0, n);
            this.fifo = this.fifo.slice(n);
            return out;
        }
    }

    if (typeof module !== 'undefined' && module.exports) {
        module.exports = SpeechStretcher;
    } else {
        root.SpeechStretcher = SpeechStretcher;
    }
})(typeof window !== 'undefined' ? window : globalThis);
