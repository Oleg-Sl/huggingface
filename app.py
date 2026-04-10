class AppBaseException(Exception):
    """Базовое исключение приложения"""
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

class AuthDataMissingError(AppBaseException):
    """Данных нет в ключнице (нужно окно ввода)"""
    pass

class InvalidCredentialsError(AppBaseException):
    """Сервер отклонил вход (неверный пароль/сертификат)"""
    pass

class TransportError(AppBaseException):
    """Сетевые проблемы (нет связи с сервером)"""
    pass

import requests

class IAuthStrategy(ABC):
    def __init__(self, base_url: str):
        self.base_url = base_url

    @abstractmethod
    def login_process(self, session: requests.Session) -> bool: pass
    
    @abstractmethod
    def get_user_path(self) -> str: pass
    
    @abstractmethod
    def get_register_path(self) -> str: pass

class LoginPasswordStrategy(IAuthStrategy):
    def __init__(self, base_url: str, username: str, storage: ISecretStorage):
        super().__init__(base_url)
        self.username = username
        self.storage = storage

    def login_process(self, session: requests.Session) -> bool:
        password = self.storage.get_password(self.username)
        if not password:
            raise AuthDataMissingError(f"Пароль для {self.username} не найден.")
        
        try:
            # Специфичный путь для логина
            session.get(f"{self.base_url}/auth/form")
            resp = session.post(f"{self.base_url}/api/v1/login", 
                                json={"u": self.username, "p": password})
            return resp.status_code == 200
        except requests.RequestException:
            raise TransportError("Ошибка сети при попытке входа")

    def get_user_path(self): return "/api/v1/profile"
    def get_register_path(self): return "/api/v1/signup"

class CertificateStrategy(IAuthStrategy):
    def __init__(self, base_url: str, cert_path: str):
        super().__init__(base_url)
        self.cert = cert_path

    def login_process(self, session: requests.Session) -> bool:
        session.cert = self.cert
        try:
            # Специфичный путь для сертификата
            resp = session.get(f"{self.base_url}/auth/cert/check")
            return resp.status_code == 200
        except requests.RequestException:
            raise TransportError("Ошибка сети (сертификат)")

    def get_user_path(self): return "/api/v2/cert-owner"
    def get_register_path(self): return "/api/v2/cert-register"

class ApiClient:
    def __init__(self, strategy: IAuthStrategy):
        self._strategy = strategy
        self._session = requests.Session()
        self._current_user = None

    def is_authorized(self) -> bool:
        """Проверка наличия загруженного пользователя"""
        return self._current_user is not None

    def login(self) -> bool:
        """Метод авторизации. Вызывает стратегию и получает данные юзера."""
        if self._strategy.login_process(self._session):
            self._current_user = self.fetch_user()
            return True
        raise InvalidCredentialsError("Сервер отклонил вход.")

    def fetch_user(self) -> dict:
        """Загрузка данных пользователя по пути из стратегии"""
        url = f"{self._strategy.base_url}{self._strategy.get_user_path()}"
        resp = self._session.get(url)
        resp.raise_for_status()
        return resp.json()

    def register(self, data: dict) -> bool:
        """Регистрация по пути из стратегии"""
        url = f"{self._strategy.base_url}{self._strategy.get_register_path()}"
        resp = self._session.post(url, json=data)
        return resp.status_code == 201

    def create_order(self, order_data: dict) -> dict:
        """Метод формирования заявки"""
        if not self.is_authorized():
            self.login()
        
        # Общий URL для всех систем
        url = f"{self._strategy.base_url}/api/orders/create"
        resp = self._session.post(url, json=order_data)
        resp.raise_for_status()
        return resp.json()


def main_app_init():
    # 1. Инициализируем хранилище
    storage = KeychainStorage(service_name="NalogService")
    
    # 2. Определяем стратегию (например, на основе конфига)
    # Здесь мы не передаем пароль, только логин
    strategy = LoginPasswordStrategy(
        base_url="https://api.nalog.ru", 
        username="oleg_dev", 
        storage=storage
    )
    
    # 3. Создаем API клиент
    client = ApiClient(strategy)

    # 4. Логика Presenter при старте
    try:
        if client.login():
            user = client.fetch_user()
            print(f"Успешный вход! Пользователь: {user.get('full_name')}")
        else:
            print("Требуется ручной ввод данных.")
            
    except AuthDataMissingError:
        print("ОКНО: Пароля нет. Пожалуйста, зарегистрируйтесь или введите пароль.")
        # Тут вызывается view.show_registration_form()
        
    except InvalidCredentialsError:
        print("ОКНО: Неверный логин или пароль.")
        
    except TransportError as e:
        print(f"ОКНО: Проблема с сетью: {e}")

if __name__ == "__main__":
    main_app_init()
