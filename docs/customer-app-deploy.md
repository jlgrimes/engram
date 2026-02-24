# Deploying `customer-app` to `app.conch.so`

This guide covers deployment and DNS routing from GoDaddy for the customer-facing app.

## Package Path

`/home/jared/projects/conch/customer-app`

## 1) Build and Smoke Test Locally

```bash
cd /home/jared/projects/conch/customer-app
npm install
npm run build
npm run start
```

Then open `http://localhost:3000`.

## 2) Choose a Deployment Target

Use one option:

### Option A: Vercel (recommended for Next.js)

1. Create/import project from `/home/jared/projects/conch/customer-app`.
2. Framework preset: `Next.js`.
3. Build command: `npm run build`.
4. Output: default Next.js output.
5. Add custom domain `app.conch.so` in Vercel project settings.

Expected GoDaddy DNS for Vercel:

| Type | Name | Value | TTL |
|---|---|---|---|
| CNAME | `app` | `cname.vercel-dns.com` | `600` (or default) |

Notes:
- Remove conflicting `A`/`CNAME` records for `app` before saving.
- If Vercel gives a project-specific target, use that exact value.

### Option B: Static host/VPS/Load Balancer

Deploy app to your host and point `app.conch.so` to a public IPv4 address.

Expected GoDaddy DNS for A-record target:

| Type | Name | Value | TTL |
|---|---|---|---|
| A | `app` | `<YOUR_PUBLIC_IPV4>` | `600` (or default) |

Optional IPv6:

| Type | Name | Value | TTL |
|---|---|---|---|
| AAAA | `app` | `<YOUR_PUBLIC_IPV6>` | `600` (or default) |

## 3) Verify DNS and Routing

Run these after saving DNS records (propagation may take several minutes):

```bash
# Check authoritative resolution
nslookup app.conch.so

# Query A record
dig +short app.conch.so A

# Query CNAME record
dig +short app.conch.so CNAME
```

HTTP checks:

```bash
# Header check
curl -I https://app.conch.so

# Follow redirects and print final URL
curl -sS -L -o /dev/null -w '%{url_effective}\n%{http_code}\n' https://app.conch.so
```

Expected result:
- DNS resolves to either the Vercel CNAME chain or your A/AAAA host.
- `https://app.conch.so` returns `200` (or expected `301/308` then `200`).

## 4) Operational Notes

- Keep internal dashboard separate: this deployment only serves `customer-app`.
- If cert issuance fails, verify no conflicting DNS entries for `app`.
- If stale DNS appears, clear local DNS cache and re-check.
