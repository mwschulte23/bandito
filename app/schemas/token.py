from pydantic import BaseModel

class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    email: str


class PasswordReset(BaseModel):
    token: str
    new_password: str