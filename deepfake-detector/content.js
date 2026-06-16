chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "ANALYSE_IMAGE") return;

  const srcUrl = msg.srcUrl;

  // Fetch the image and convert to dataURL
  fetch(srcUrl)
    .then(r => r.blob())
    .then(blob => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    }))
    .then(dataUrl => {
      chrome.runtime.sendMessage({ type: "OPEN_ANALYSIS", dataUrl });
    })
    .catch(err => {
      console.error("DeepfakeShield: Failed to fetch image", err);
      alert("DeepfakeShield: Could not load this image. It may be cross-origin protected.");
    });
});