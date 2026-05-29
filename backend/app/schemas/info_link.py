from pydantic import BaseModel, Field


class InfoLink(BaseModel):
    label: str
    url: str


class InfoLinkGroup(BaseModel):
    id: str
    title: str
    links: list[InfoLink] = Field(default_factory=list)


class InfoLinkListResponse(BaseModel):
    groups: list[InfoLinkGroup] = Field(default_factory=list)
