const DEFAULT_BACKEND_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_BACKEND_INFO_LINKS_API_PATH = "/api/v1/extra/info-links";

const backendApiBaseUrl =
  process.env.BACKEND_API_BASE_URL || DEFAULT_BACKEND_API_BASE_URL;
const backendInfoLinksApiPath =
  process.env.BACKEND_INFO_LINKS_API_PATH || DEFAULT_BACKEND_INFO_LINKS_API_PATH;

const createBackendInfoLinksUrl = () =>
  new URL(backendInfoLinksApiPath, backendApiBaseUrl).toString();

export async function GET() {
  try {
    const backendResponse = await fetch(createBackendInfoLinksUrl(), {
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    });

    if (!backendResponse.ok) {
      return Response.json(
        { groups: [] },
        { status: backendResponse.status },
      );
    }

    return Response.json(await backendResponse.json());
  } catch {
    return Response.json({ groups: [] }, { status: 502 });
  }
}
