import json
import random

users = [
    "Нет учётной записи в СЭД ТУСУР",
    "Ошибка входа в систему",
    "Требуется доступ к корпоративной почте",
    "Не удаётся установить ПО MathCad",
    "Проблемы с подключением к VPN",
    "Не работает электронная подпись",
    "Не могу скачать файлы с портала",
    "Требуется продление лицензии Windows",
    "Возникает ошибка 1004 при входе",
    "Потерян доступ к ФДО ТУСУР",
    "Забыли пароль от MS Office",
    "Проблема с активацией Visual Studio",
    "Требуется подключение к удалённому рабочему столу",
    "Не отображается курс в ФДО",
    "Ошибка при отправке документов",
    "Вопрос по тарифу ПО",
    "Сбой работы Outlook",
    "Запросить доступ к базе данных",
    "Установить новый принтер",
    "Запросить сертификат безопасности",
    "Проблема с авторизацией на сайте",
    "Требуется доступ к Zoom",
    "Ошибка обновления ПО",
    "Требуется помощь с настройкой роутера",
    "Не работает загрузка файлов",
    "Проблемы с загрузкой домашнего задания",
    "Проблема с лицензией на MathCad",
    "Требуется новый аккаунт в системе",
    "Перенос почты на другой сервер",
    "Возникает ошибка при запуске 1С",
    "Не приходит письмо с подтверждением",
    "Проблема с подключением к Wi-Fi",
    "Ошибка синхронизации календаря",
    "Запрос на восстановление данных",
    "Не открывается учебный материал",
    "Требуется перенос данных между аккаунтами",
    "Проблема с PDF-документом",
    "Ошибка загрузки результатов экзамена",
    "Проблема с доступом к облаку",
    "Не удаётся активировать продукт",
    "Вопрос по политике безопасности",
    "Требуется консультация по ПО",
    "Не работает кнопка отправки",
    "Ошибка доступа к серверу",
    "Сбросить настройки безопасности",
    "Проблема с получением SMS-кода",
    "Вопрос по работе портала",
    "Требуется замена оборудования",
    "Ошибка при установке драйвера",
]

answers = [
    "Учетная запись создана, письмо с учетными данными выслано на почту: [EMAIL]",
    "Ошибка устранена, попробуйте снова войти.",
    "Доступ к почте предоставлен. Инструкция выслана на почту.",
    "ПО MathCad установлено. Проверьте работоспособность.",
    "Доступ к VPN открыт. Проверьте параметры подключения.",
    "Настроена электронная подпись, можно пользоваться.",
    "Возможность скачивания файлов восстановлена.",
    "Лицензия Windows продлена на 1 год.",
    "Ошибка 1004 исправлена, повторите попытку.",
    "Доступ к ФДО восстановлен.",
    "Пароль для MS Office отправлен на указанную почту.",
    "Visual Studio активирована.",
    "Доступ к удалённому рабочему столу предоставлен.",
    "Курс теперь отображается в вашем аккаунте.",
    "Отправка документов успешно настроена.",
    "Вопрос по тарифу обработан, подробности отправлены.",
    "Работа Outlook восстановлена.",
    "Доступ к базе данных открыт.",
    "Новый принтер установлен и готов к работе.",
    "Сертификат безопасности выдан.",
    "Авторизация на сайте восстановлена.",
    "Доступ к Zoom предоставлен.",
    "ПО успешно обновлено.",
    "Роутер настроен, доступ к сети есть.",
    "Загрузка файлов теперь работает корректно.",
    "Загрузка домашнего задания восстановлена.",
    "Лицензия на MathCad продлена.",
    "Аккаунт в системе создан, детали на почте.",
    "Перенос почты выполнен.",
    "Ошибка при запуске 1С устранена.",
    "Письмо с подтверждением отправлено повторно.",
    "Wi-Fi подключение восстановлено.",
    "Синхронизация календаря работает.",
    "Данные восстановлены.",
    "Учебный материал доступен для просмотра.",
    "Данные перенесены между аккаунтами.",
    "PDF-документ открыт без ошибок.",
    "Результаты экзамена успешно загружены.",
    "Доступ к облаку предоставлен.",
    "Продукт активирован.",
    "Информация по политике безопасности отправлена.",
    "Консультация по ПО проведена.",
    "Кнопка отправки работает.",
    "Доступ к серверу восстановлен.",
    "Настройки безопасности сброшены.",
    "SMS-код отправлен повторно.",
    "Вопрос по порталу обработан.",
    "Оборудование заменено.",
    "Драйвер установлен успешно.",
]

categories = [
    "Электронный документооборот",
    "1С",
    "Техническая проблема на рабочем месте сотрудника",
    "Автоматизированные информационные системы ТУСУРа",
    "Не могу определитьНичего не подходит",
    "Антиплагиат.ВУЗ",
    "Получение лицензионного ПО",
    "Получение корпоративной почты",
    "Веб-ресурсы ТУСУР",
    "Заявка на размещение публикаций",
    "Предложения изменений в информационные системы",
    "Мобильные приложения ТУСУР",
]

testset = []
for i in range(500):
    user_msg = random.choice(users)
    answer = random.choice(answers)
    cat = random.choice(categories)
    ticket_id = 7000 + i  # Можно делать уникальные ID

    testset.append(
        {
            "text": f"Категория: {cat}\nПользователь: {user_msg}\nОператор: ",
            "label": answer,
            "category_title": cat,
            "ticket_id": ticket_id,
        }
    )

# Сохраняем в файл (если нужно)
with open("testset.json", "w", encoding="utf-8") as f:
    json.dump(testset, f, ensure_ascii=False, indent=2)

print("Готово! 500 синтетических тестовых объектов сгенерировано.")
