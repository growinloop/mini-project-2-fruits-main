# **코딩 컨벤션(Coding Convention)**

**작 성 자:** 팀 Fruits **작 성 일:** 2025-11-11

## **1. 명 명 규 칙 (Naming Rules)**

일관성 있는 네이밍은 코드의 가독성을 높이는 핵심 요소입니다.

### **1.1. 디렉토리 및 패키지 (Directories & Packages)**

-   **패키지 (디렉토리):** 짧고, 의미를 알 수 있는 **소문자**로 작성합니다. 밑줄(\_) 사용은 지양합니다.

    -   *Ex:* src, models, data

-   **모듈 (파일):** **전체 소문자**를 사용하며, 필요한 경우 단어 사이에 밑줄(\_)을 넣어 가독성을 높입니다. (snake_case)

    -   *Ex:* data_loader.py, train_model.py

### **1.2. 클래스 (Classes)**

-   각 단어의 첫 글자를 대문자로 하는 **CapWords** (PascalCase) 방식을 사용합니다.

    -   *Ex:* ImageClassifier, DatabaseConnection

-   내부용(internal) 클래스는 앞에 밑줄(\_) 하나를 붙일 수 있습니다.

    -   *Ex:* \_InternalHelper

### **1.3. 필드 (Fields)**

-   인스턴스 변수(필드)는 **lowercase_with_underscores** (snake_case) 방식을 사용합니다.

    -   *Ex:* self.image_size, self.batch_size

-   \'protected\' 멤버(하위 클래스 접근 허용)는 앞에 밑줄(\_) 하나를 붙입니다.

    -   *Ex:* self.\_device

-   \'private\' 멤버(클래스 내부에서만 사용)는 앞에 밑줄(\_\_) 두 개를 붙입니다.

    -   *Ex:* self.\_\_internal_state

### **1.4. 메소드 (Methods)**

-   함수 및 메소드 이름은 **lowercase_with_underscores** (snake_case) 방식을 사용합니다.

    -   *Ex:* preprocess_data, calculate_iou

-   \'protected\'/\'private\' 규칙은 필드와 동일하게 적용합니다.

    -   *Ex:* self.\_load_weights, self.\_\_validate_input

### **1.5. 변수 및 상수 (Variables & Constants)**

-   **변수:** lowercase_with_underscores (snake_case) 방식을 사용합니다.

    -   *Ex:* img_path, total_loss, user_name

-   **상수:** 모듈 레벨의 상수는 **ALL_CAPS_WITH_UNDERSCORES** 방식을 사용합니다.

    -   *Ex:* BASE_DIR, IMG_SIZE = 512, MAX_EPOCHS

### **1.6. 객체 (Objects)**

-   객체 인스턴스를 가리키는 변수명은 명확하고, lowercase_with_underscores 규칙을 따릅니다.

-   이름은 객체의 역할이나 내용을 명확히 나타내야 합니다.

    -   *Ex:* model = YOLO(), train_loader = DataLoader(\...)

-   일반적인 이름(e.g., d, o, data)의 사용은 짧은 범위(e.g., 람다, 짧은 루프)에서만 제한적으로 허용합니다.

### **1.7. 데이터베이스 (Database)**

-   **테이블:** lowercase_with_underscores (snake_case)를 사용하며, 복수형을 권장합니다.

    -   *Ex:* users, products, order_details

-   **컬럼:** lowercase_with_underscores (snake_case)를 사용합니다.

    -   *Ex:* user_id, created_at, product_name

-   **기본 키 (PK):** id 또는 (테이블명)\_id (예: id, user_id)

-   **외래 키 (FK):** 참조하는 테이블의 PK를 따릅니다. (예: user_id가 users 테이블의 id를 참조)

## **2. 작 성 규 칙 (Writing Rules)**

### **2.1. 들여쓰기 (Indentation)**

-   들여쓰기는 **스페이스(Space) 4개**를 사용합니다.

-   탭(Tab) 문자는 사용하지 않습니다.

### **2.2. 바디 (Body)**

#### **2.2.1. 줄 길이 (Line Length)**

-   한 줄의 최대 길이는 **79자** (PEP 8 표준) 또는 팀 합의하에 최대 **100자**를 넘지 않도록 합니다.

-   긴 줄은 괄호((), {}, \[\]) 안에서 줄 바꿈을 하거나, 역슬래시(\\)를 사용하여 나눕니다. (괄호 안에서의 줄 바꿈을 권장)

#### **2.2.2. 빈 줄 (Blank Lines)**

-   최상위 함수와 클래스 정의는 2줄의 빈 줄로 구분합니다.

-   클래스 내의 메소드 정의는 1줄의 빈 줄로 구분합니다.

-   함수/메소드 내에서는 논리적인 구분을 위해 1줄의 빈 줄을 적절히 사용할 수 있습니다.

#### **2.2.3. 공백 (Spacing)**

-   이항 연산자(e.g., =, +, -, ==, \>)의 앞뒤에는 1개의 공백을 둡니다.

-   쉼표(,) 뒤에는 항상 공백을 둡니다. (예: \[1, 2, 3\])

-   키워드 인자(e.g., func(name=\"Gildong\"))의 = 주위에는 공백을 두지 않습니다.

-   주석 표기 (#) 뒤에는 1개의 공백을 둡니다. (예: \# 이것은 주석입니다.)

#### **2.2.4. 임포트 (Imports)**

-   모든 import 문은 파일의 맨 위에 위치합니다.

-   다음 순서로 그룹화하고, 그룹 사이에는 빈 줄을 둡니다.

    1.  표준 라이브러리 (e.g., os, json)

    2.  서드파티 라이브러리 (e.g., torch, numpy, ultralytics)

    3.  로컬(자체 제작) 라이브러리 (e.g., from .utils import helper)

-   import \* (Wildcard imports)는 사용하지 않습니다.

#### **2.2.5. 문자열 (Strings)**

-   따옴표는 일관되게 사용합니다. (작은따옴표 \'\' 또는 큰따옴표 \"\" 중 하나로 통일)

-   문자열 포맷팅은 f-string 사용을 최우선으로 권장합니다.

    -   *Good:* f\"User {user_name} (ID: {user_id})\"

    -   *Avoid:* User \" + user_name + \" (ID: \" + str(user_id) + \")\"


---