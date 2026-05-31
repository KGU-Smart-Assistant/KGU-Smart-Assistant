const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const infoLinksApiPath = "/api/info-links";

const createInfoLinksApiUrl = () =>
  apiBaseUrl ? new URL(infoLinksApiPath, apiBaseUrl).toString() : infoLinksApiPath;

export async function fetchInfoLinkGroups() {
  const response = await fetch(createInfoLinksApiUrl(), {
    headers: {
      Accept: "application/json",
    },
    cache: "no-store",
  });

  if (!response.ok) {
    throw new Error(`Info link request failed with status ${response.status}`);
  }

  const payload = await response.json();
  return Array.isArray(payload.groups) ? payload.groups : [];
}
