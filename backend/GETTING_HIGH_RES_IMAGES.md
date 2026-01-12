# How to Get Free High-Resolution Arctic Satellite Images
## (Sentinel-1 & Sentinel-2 via Copernicus Data Space Ecosystem)

To enhance your Arctic Ice Monitoring application with high-resolution imagery and precise ice polygons, you need access to the **Copernicus Data Space Ecosystem (CDSE)**. This replaces the old "SciHub" which is now retired.

**Resolution:**
- **Sentinel-1 (SAR):** ~10m resolution (Excellent for ice/water distinction, works through clouds/night).
- **Sentinel-2 (Optical):** ~10m resolution (Visual colors, good for validation, but blocked by clouds).

### Step 1: Register for a Free Account

1.  Go to **[dataspace.copernicus.eu](https://dataspace.copernicus.eu/)**.
2.  Click **"Register"** or **"Login"** (top right).
3.  Create a new account (it's completely free for personal and commercial use).
4.  Verify your email.

### Step 2: Get OAuth API Credentials (Recommended)

1.  Log in to the **[CDSE Dashboard](https://dataspace.copernicus.eu/browser/)**.
2.  Go to **Settings** (or User Settings).
3.  Look for **"OAuth Client"**.
4.  Click **"Create OAuth Client"**.
5.  Select **"Client Credentials"** grant type (if asked).
6.  Copy the **Client ID** and **Client Secret**.

### Step 3: Configure Your Application

Update your `.env` file in the project root with the new credentials:

```ini
# Copernicus Data Space Ecosystem (CDSE) Credentials
CDSE_CLIENT_ID=your_client_id
CDSE_CLIENT_SECRET=your_client_secret
```

### Step 4: Verify Access

You can manually verify access by browsing the **[Copernicus Browser](https://browser.dataspace.copernicus.eu/)**:
1.  Zoom to the Arctic.
2.  Set the date range to "Today".
3.  Look for **Sentinel-1** (SAR - grayscale) stripes over the ocean. These are the high-res images our app will process.
4.  Look for **Sentinel-2** (Optical - color) tiles (might be cloudy).

### Cost
- **License:** Open data policy (Free).
- **Quotas:** Generous limits for standard users. More than enough for daily Arctic monitoring.

---

**Next Steps:**
Once you have these credentials and updated the `.env` file, let me know, and I will proceed with **updating the backend** to fetch this data and generate the precise ice polygons you requested.
