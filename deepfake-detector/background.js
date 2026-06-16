chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "checkDeepfake",
    title: "🔍 Check for Deepfake",
    contexts: ["image"]
  });
  console.log("DeepfakeShield: installed");
});

// Service workers can fetch cross-origin images (host_permissions: <all_urls>)
// Converting to data URL here avoids CORS errors in the analysis page
async function urlToDataUrl(url) {
  const response = await fetch(url);
  const arrayBuffer = await response.arrayBuffer();
  const mimeType = response.headers.get('content-type') || 'image/jpeg';
  const uint8 = new Uint8Array(arrayBuffer);
  let binary = '';
  const chunkSize = 8192;
  for (let i = 0; i < uint8.length; i += chunkSize) {
    binary += String.fromCharCode(...uint8.subarray(i, i + chunkSize));
  }
  return `data:${mimeType};base64,${btoa(binary)}`;
}

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "checkDeepfake") return;
  console.log("DeepfakeShield: clicked", info.srcUrl);

  let imageData = info.srcUrl;
  try {
    imageData = await urlToDataUrl(info.srcUrl);
    console.log("DeepfakeShield: converted to data URL, length:", imageData.length);
  } catch (e) {
    console.warn("DeepfakeShield: fetch failed, falling back to raw URL:", e.message);
  }

  chrome.storage.local.set({ pendingImage: imageData }, () => {
    chrome.windows.create({
      url: chrome.runtime.getURL("analysis.html"),
      type: "popup",
      width: 480,
      height: 780,
      focused: true
    });
  });
});