// Generated from the existing I Am identity by tools/sync-universal-one-character.py.
(()=>{
  function makeMiniPlanetTexture(seed) {
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    const ocean = context.createLinearGradient(0, 0, canvas.width, canvas.height);
    ocean.addColorStop(0, seed % 2 ? "#0b4d72" : "#0d6380");
    ocean.addColorStop(1, "#071f42");
    context.fillStyle = ocean;
    context.fillRect(0, 0, canvas.width, canvas.height);
    const islands = [
      [.08, .32, .2, .19],
      [.32, .56, .26, .24],
      [.59, .24, .22, .22],
      [.76, .62, .28, .2],
    ];
    context.fillStyle = seed % 2 ? "#4a8a55" : "#3d9a70";
    islands.forEach(([x, y, width, height], index) => {
      context.beginPath();
      context.ellipse(((x + seed * .07) % 1) * canvas.width, y * canvas.height, width * canvas.width, height * canvas.height, index * .42, 0, Math.PI * 2);
      context.fill();
    });
    context.strokeStyle = "rgba(228, 190, 123, 0.62)";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(0, canvas.height * .48);
    context.bezierCurveTo(canvas.width * .27, canvas.height * .32, canvas.width * .63, canvas.height * .66, canvas.width, canvas.height * .44);
    context.stroke();
    return canvas.toDataURL("image/png");
  }

window.UniversalOneIdentity={texture:makeMiniPlanetTexture,character:"    <div class=\"iam-realtime-character\" aria-hidden=\"true\">\n      <span class=\"iam-realtime-character__shadow\"></span>\n      <span class=\"iam-realtime-character__head\">\n        <span class=\"iam-realtime-character__visor\">\n          <i class=\"iam-realtime-character__eye iam-realtime-character__eye--left\"></i>\n          <i class=\"iam-realtime-character__eye iam-realtime-character__eye--right\"></i>\n        </span>\n        <span class=\"iam-realtime-character__headphone iam-realtime-character__headphone--left\"></span>\n        <span class=\"iam-realtime-character__headphone iam-realtime-character__headphone--right\"></span>\n        <span class=\"iam-realtime-character__headband\"></span>\n      </span>\n      <span class=\"iam-realtime-character__body\">\n        <img class=\"iam-realtime-character__flower\" src=\"/vogue/iam/favicon.png?v=iam-favicon-4\" alt=\"\">\n      </span>\n      <span class=\"iam-realtime-character__arm iam-realtime-character__arm--left\"><i class=\"iam-realtime-character__hand\"></i></span>\n      <span class=\"iam-realtime-character__arm iam-realtime-character__arm--right\"><i class=\"iam-realtime-character__hand\"></i></span>\n      <span class=\"iam-realtime-character__leg iam-realtime-character__leg--left\"><i></i></span>\n      <span class=\"iam-realtime-character__leg iam-realtime-character__leg--right\"><i></i></span>\n      <span class=\"iam-realtime-character__tablet\"><i></i><b></b><em></em></span>\n      <span class=\"iam-realtime-character__tea\"><i></i><b></b><em></em></span>\n      <span class=\"iam-realtime-character__speaker iam-realtime-character__speaker--left\"><i></i></span>\n      <span class=\"iam-realtime-character__speaker iam-realtime-character__speaker--right\"><i></i></span>\n    </div>\n"};
})();
