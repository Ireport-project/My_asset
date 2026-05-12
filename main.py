"""config.yaml용 bcrypt 비밀번호 해시 생성기.

streamlit-authenticator가 credentials에 저장하는 형식과 동일하게 해시합니다.
생성된 문자열을 config.yaml의 `password:` 값에 붙여 넣으세요.

사용 예:
    python hash_password_for_config.py
    python hash_password_for_config.py --password "새비밀번호"

주의: 터미널에 평문 비밀번호를 넘기면 셸 기록에 남을 수 있습니다.
"""

from __future__ import annotations

import argparse
import getpass
import sys

import streamlit_authenticator as stauth


def main() -> int:
    parser = argparse.ArgumentParser(
        description="config.yaml의 password 필드에 넣을 bcrypt 해시를 출력합니다."
    )
    parser.add_argument(
        "-p",
        "--password",
        help="평문 비밀번호 (생략 시 입력 프롬프트, 에코 없음)",
    )
    args = parser.parse_args()

    if args.password is not None:
        plain = args.password
    else:
        plain = getpass.getpass("비밀번호: ")
        confirm = getpass.getpass("비밀번호 확인: ")
        if plain != confirm:
            print("오류: 두 입력이 일치하지 않습니다.", file=sys.stderr)
            return 1

    if not plain:
        print("오류: 비밀번호가 비어 있습니다.", file=sys.stderr)
        return 1

    hashed = stauth.Hasher().hash(plain)

    print()
    print("# config.yaml 예시 (들여쓰기는 사용자명 블록에 맞게 조정)")
    print(f'password: "{hashed}"')
    print()
    print("또는 YAML 단일 인용부호로 감싸도 됩니다:")
    print(f"password: '{hashed}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
