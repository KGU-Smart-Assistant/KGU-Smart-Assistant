const DEFAULT_BACKEND_API_BASE_URL = "http://127.0.0.1:8000";
const DEFAULT_BACKEND_CONTACTS_API_PATH = "/api/v1/extra/contacts";

const backendApiBaseUrl =
  process.env.BACKEND_API_BASE_URL || DEFAULT_BACKEND_API_BASE_URL;
const backendContactsApiPath =
  process.env.BACKEND_CONTACTS_API_PATH || DEFAULT_BACKEND_CONTACTS_API_PATH;

const createBackendContactsUrl = () =>
  new URL(backendContactsApiPath, backendApiBaseUrl).toString();

export async function GET() {
  try {
    const backendResponse = await fetch(createBackendContactsUrl(), {
      headers: {
        Accept: "application/json",
      },
      cache: "no-store",
    });

    if (!backendResponse.ok) {
      return Response.json(
        { contacts: [] },
        { status: backendResponse.status },
      );
    }

    return Response.json(await backendResponse.json());
  } catch (error) {
    return Response.json({ contacts: [] }, { status: 502 });
  }
}
