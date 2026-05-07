from pydantic import BaseModel, Field


class DepartmentContact(BaseModel):
    department_id: str
    department_name: str
    phone_number: str
    description: str | None = None
    tel_uri: str = Field(description="URI that Android can open with ACTION_DIAL")


class DepartmentContactListResponse(BaseModel):
    contacts: list[DepartmentContact]
