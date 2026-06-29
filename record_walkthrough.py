"""Record one full UNVEIL walkthrough cycle.

Launches Chromium via Playwright, loads
http://localhost:8001/unveil-visualization.html, lets the page play for the
duration of one full sequence (3.1 → ... → human reveal), then closes so the
video is finalized.

Output: ./videos/unveil-walkthrough-<timestamp>.webm
"""
import asyncio
import os
import shutil
import time
from playwright.async_api import async_playwright

URL = "http://localhost:8001/unveil-visualization.html?nopoll=1"

# Stage durations from defineStages (must match the JS):
#   robot 1.0  +  kinematic 7.2  +  enc-output-pause 1.0  +
#   graph 29.2 +  temporal 11.5  +  embedding 3.0 +  attributes 3.0 +
#   human-reveal 1.0  +  small buffer
RECORD_TIMEOUT_S = 600    # hard cap if the walkthrough hangs; normally finishes much sooner

VIDEO_W = 1720
VIDEO_H = 820

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
os.makedirs(OUT_DIR, exist_ok=True)


async def main():
    async with async_playwright() as p:
        # Disable background/throttling flags so requestAnimationFrame fires
        # at full ~60 fps even though the headless Chromium tab is "occluded".
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=CalculateNativeWinOcclusion",
            ],
        )
        context = await browser.new_context(
            viewport={"width": VIDEO_W, "height": VIDEO_H},
            record_video_dir=OUT_DIR,
            record_video_size={"width": VIDEO_W, "height": VIDEO_H},
            device_scale_factor=2,
        )
        page = await context.new_page()
        print(f"Loading {URL} ...")
        await page.goto(URL, wait_until="load")
        # Wait for the loading overlay to disappear (boot finished).
        try:
            await page.wait_for_function(
                "document.getElementById('loading') && "
                "document.getElementById('loading').style.display === 'none'",
                timeout=30000,
            )
        except Exception:
            pass
        # Brief settle, then record until the walkthrough completes — the
        # Replay button receives the 'visible' class inside endWalkthrough(),
        # which is exactly when the human-reveal stage has finished. We close
        # the context shortly after so the auto-replay doesn't bleed into the
        # video.
        await page.wait_for_timeout(500)
        print("Recording until walkthrough finishes ...")
        await page.wait_for_function(
            "document.getElementById('replay-btn') && "
            "document.getElementById('replay-btn').classList.contains('visible')",
            timeout=RECORD_TIMEOUT_S * 1000,
        )
        # Small buffer so the final frame (human visible) is in the video.
        await page.wait_for_timeout(800)
        # Save the video by closing the context.
        await context.close()
        await browser.close()

    # Rename the auto-generated file to a friendlier name with timestamp.
    files = [f for f in os.listdir(OUT_DIR) if f.endswith(".webm")]
    if files:
        latest = max(files, key=lambda f: os.path.getmtime(os.path.join(OUT_DIR, f)))
        ts = time.strftime("%Y%m%d-%H%M%S")
        new_name = f"unveil-walkthrough-{ts}.webm"
        shutil.move(os.path.join(OUT_DIR, latest), os.path.join(OUT_DIR, new_name))
        print(f"Saved: {os.path.join(OUT_DIR, new_name)}")
    else:
        print("No video file produced.")


if __name__ == "__main__":
    asyncio.run(main())
