const DEFAULT_CONTACTS_API_PATH = "/api/contacts";

const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL;
const contactsApiPath =
  process.env.NEXT_PUBLIC_CONTACTS_API_PATH || DEFAULT_CONTACTS_API_PATH;

const createContactsUrl = () =>
  apiBaseUrl
    ? new URL(contactsApiPath, apiBaseUrl).toString()
    : contactsApiPath;

const inferCampus = (contact) => {
  const text = `${contact.department_name ?? ""} ${contact.description ?? ""}`;

  return text.includes("서울") ? "서울캠퍼스" : "수원캠퍼스";
};

const inferCategory = (contact) => {
  const text = `${contact.department_name ?? ""} ${contact.description ?? ""}`;

  if (text.includes("입학") || text.includes("모집") || text.includes("원서")) {
    return "입학";
  }

  if (
    text.includes("학사") ||
    text.includes("수업") ||
    text.includes("졸업") ||
    text.includes("성적")
  ) {
    return "학사";
  }

  if (
    text.includes("학생") ||
    text.includes("장학") ||
    text.includes("복지") ||
    text.includes("도서관")
  ) {
    return "학생지원";
  }

  if (text.includes("시설") || text.includes("건물") || text.includes("설비")) {
    return "시설관리";
  }

  if (text.includes("대표") || text.includes("안내")) {
    return "대표번호";
  }

  return "행정";
};

const createDepartmentId = (contact, index) => {
  const seed =
    contact.department_id ??
    `${contact.department_name ?? "contact"}-${contact.phone_number ?? index}`;

  return `api-contact-${String(seed)
    .trim()
    .replace(/\s+/g, "-")
    .replace(/[^0-9A-Za-z가-힣_-]/g, "")}`;
};

const normalizeContact = (contact, index) => {
  const name = contact.department_name ?? "";
  const phone = contact.phone_number ?? "";

  if (!name || !phone) {
    return null;
  }

  return {
    campus: inferCampus(contact),
    category: inferCategory(contact),
    department: {
      id: createDepartmentId(contact, index),
      name,
      phone,
      description: contact.description ?? "백엔드 연락처 데이터",
      telUri: contact.tel_uri,
    },
  };
};

const groupContactsBySection = (contacts) => {
  const sectionMap = new Map();

  contacts.forEach((contact, index) => {
    const normalized = normalizeContact(contact, index);

    if (!normalized) {
      return;
    }

    const key = `${normalized.campus}__${normalized.category}`;
    const section = sectionMap.get(key) ?? {
      campus: normalized.campus,
      category: normalized.category,
      departments: [],
    };

    section.departments.push(normalized.department);
    sectionMap.set(key, section);
  });

  return Array.from(sectionMap.values());
};

export const mergePhoneDirectories = (baseDirectory, apiDirectory) => {
  if (!apiDirectory.length) {
    return baseDirectory;
  }

  const mergedMap = new Map(
    baseDirectory.map((section) => [
      `${section.campus}__${section.category}`,
      {
        ...section,
        departments: [...section.departments],
      },
    ]),
  );

  apiDirectory.forEach((apiSection) => {
    const key = `${apiSection.campus}__${apiSection.category}`;
    const section = mergedMap.get(key) ?? {
      campus: apiSection.campus,
      category: apiSection.category,
      departments: [],
    };
    const existingKeys = new Set(
      section.departments.map(
        (department) => `${department.name}__${department.phone}`,
      ),
    );

    apiSection.departments.forEach((department) => {
      const departmentKey = `${department.name}__${department.phone}`;

      if (!existingKeys.has(departmentKey)) {
        section.departments.push(department);
        existingKeys.add(departmentKey);
      }
    });

    mergedMap.set(key, section);
  });

  return Array.from(mergedMap.values());
};

export async function requestPhoneDirectory({ signal } = {}) {
  const response = await fetch(createContactsUrl(), {
    headers: {
      Accept: "application/json",
    },
    signal,
  });

  if (!response.ok) {
    throw new Error(`Contacts API request failed: ${response.status}`);
  }

  const data = await response.json();
  const contacts = Array.isArray(data?.contacts) ? data.contacts : [];

  return groupContactsBySection(contacts);
}
