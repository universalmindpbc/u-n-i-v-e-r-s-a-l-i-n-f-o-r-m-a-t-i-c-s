
      (() => {
        const canvas = document.querySelector("[data-rain-canvas]");
        const context = canvas?.getContext("2d", { alpha: false });
        if (!canvas || !context) return;

        const motionQuery = window.matchMedia("(prefers-reduced-motion: reduce)");
        const abstractGlyphs = ["0", "1", "1", "2", "3", "5", "8", "13", "21", "34", "55", "89", "144", "233", "377"];
        const signatureGlyphs = ["φ", "φ", "Φ"];
        const pearlTones = [
          [255, 245, 237],
          [255, 224, 236],
          [232, 226, 255]
        ];
        const state = {
          width: 0,
          height: 0,
          ratio: 1,
          cell: 22,
          columns: [],
          frame: 0,
          lastTime: 0,
          exitingAt: 0
        };

        const randomBetween = (minimum, maximum) => minimum + Math.random() * (maximum - minimum);
        const pick = (items) => items[Math.floor(Math.random() * items.length)];
        const nextGlyph = () => {
          const character = Math.random() < 0.18 ? pick(signatureGlyphs) : pick(abstractGlyphs);
          const anomaly = (character === "φ" && Math.random() < 0.22) || Math.random() < 0.026;
          return {
            character,
            editorial: Math.random() < 0.17,
            anomaly,
            tone: anomaly ? pick(pearlTones) : [255, 255, 255]
          };
        };

        const createColumn = (index) => {
          const trailLength = Math.round(randomBetween(18, 36));
          return {
            x: index * state.cell * 2 + state.cell * randomBetween(0.32, 0.68),
            headY: randomBetween(state.cell, state.height + (trailLength - 2) * state.cell),
            speed: randomBetween(66, 152),
            opacity: randomBetween(0.7, 1),
            trailLength,
            advance: randomBetween(0, state.cell),
            glyphs: Array.from({ length: trailLength }, nextGlyph)
          };
        };

        const rebuild = () => {
          state.width = Math.max(1, window.innerWidth);
          state.height = Math.max(1, window.innerHeight);
          state.ratio = Math.min(window.devicePixelRatio || 1, 1.75);
          state.cell = state.width < 520 ? 18 : state.width < 980 ? 21 : 23;

          canvas.width = Math.round(state.width * state.ratio);
          canvas.height = Math.round(state.height * state.ratio);
          canvas.style.width = `${state.width}px`;
          canvas.style.height = `${state.height}px`;
          context.setTransform(state.ratio, 0, 0, state.ratio, 0, 0);

          const columnCount = Math.ceil(state.width / (state.cell * 2)) + 1;
          state.columns = Array.from({ length: columnCount }, (_, index) => createColumn(index));
          canvas.dataset.rainColumns = String(columnCount);
          canvas.dataset.rainPalette = "monochrome";
          canvas.dataset.rainDensity = "full";
          canvas.dataset.rainAnomalies = "pearl-blush";
          canvas.dataset.rainStatus = motionQuery.matches ? "static" : "running";
        };

        const recycleColumn = (column) => {
          column.headY = randomBetween(-state.cell * 5, -state.cell);
          column.speed = randomBetween(66, 152);
          column.opacity = randomBetween(0.7, 1);
          column.trailLength = Math.round(randomBetween(18, 36));
          column.advance = 0;
          column.glyphs = Array.from({ length: column.trailLength }, nextGlyph);
        };

        const draw = (time) => {
          const elapsed = state.lastTime ? Math.min((time - state.lastTime) / 1000, 0.08) : 0;
          state.lastTime = time;
          const departure = state.exitingAt ? Math.min((time - state.exitingAt) / 700, 1) : 0;
          const speedMultiplier = 1 + departure * 5;

          context.fillStyle = "#000";
          context.fillRect(0, 0, state.width, state.height);
          context.textAlign = "center";
          context.textBaseline = "middle";

          state.columns.forEach((column) => {
            if (!motionQuery.matches) {
              column.headY += column.speed * elapsed * speedMultiplier;
              column.advance += column.speed * elapsed * speedMultiplier;
              while (column.advance >= state.cell) {
                column.advance -= state.cell;
                column.glyphs.unshift(nextGlyph());
                column.glyphs.pop();
              }
            }

            const direction = column.x < state.width / 2 ? -1 : 1;
            const departureOffset = direction * departure * departure * state.width * 0.11;

            for (let index = column.trailLength - 1; index >= 0; index -= 1) {
              const y = column.headY - index * state.cell;
              if (y < -state.cell || y > state.height + state.cell) continue;

              const glyph = column.glyphs[index];
              const distance = index / Math.max(column.trailLength - 1, 1);
              const alpha = Math.max(0.035, Math.pow(1 - distance, 1.42) * column.opacity);
              const isHead = index === 0;
              const isSignatureV = glyph.character === "φ" && glyph.editorial;
              const [red, green, blue] = glyph.tone;

              context.font = isSignatureV
                ? `${Math.round(state.cell * 1.06)}px Didot, "Bodoni 72", "Times New Roman", serif`
                : `${Math.round(state.cell * 0.86)}px "SFMono-Regular", Menlo, Consolas, monospace`;
              context.fillStyle = `rgba(${red}, ${green}, ${blue}, ${isHead ? Math.min(1, alpha + 0.22) : alpha})`;
              context.shadowColor = isHead
                ? `rgba(${red}, ${green}, ${blue}, ${glyph.anomaly ? 0.98 : 0.92})`
                : "transparent";
              context.shadowBlur = isHead ? 11 : 0;
              context.fillText(glyph.character, column.x + departureOffset, y);
            }

            if (column.headY - column.trailLength * state.cell > state.height) {
              recycleColumn(column);
            }
          });

          context.shadowBlur = 0;
          state.frame = window.requestAnimationFrame(render);
        };

        const render = (time) => {
          if (document.hidden || document.body.dataset.entered === "true") return;
          if (time - state.lastTime < 1000 / 30 && state.lastTime) {
            state.frame = window.requestAnimationFrame(render);
            return;
          }
          draw(time);
        };

        const start = () => {
          window.cancelAnimationFrame(state.frame);
          state.lastTime = 0;
          if (motionQuery.matches) {
            draw(performance.now());
            window.cancelAnimationFrame(state.frame);
            state.frame = 0;
            return;
          }
          state.frame = window.requestAnimationFrame(render);
        };

        const handleMotionChange = () => {
          canvas.dataset.rainStatus = motionQuery.matches ? "static" : "running";
          start();
        };

        rebuild();
        start();
        window.addEventListener("resize", () => {
          rebuild();
          start();
        }, { passive: true });
        document.addEventListener("visibilitychange", () => {
          if (!document.hidden) start();
        });
        document.addEventListener("informatics:return", start);
        motionQuery.addEventListener?.("change", handleMotionChange);
      })();
    