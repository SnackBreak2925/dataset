# Разница между JOIN (INNER JOIN) и OUTER JOIN с примерами

Основное различие между обычным JOIN (который является INNER JOIN) и OUTER JOIN заключается в том, какие строки включаются в результат при отсутствии соответствий между таблицами.

## INNER JOIN (просто JOIN)

**Возвращает только те строки, где есть совпадение в обеих таблицах.**

Пример:
```sql
SELECT employees.name, departments.name
FROM employees
INNER JOIN departments ON employees.department_id = departments.id;
```
Результат: только сотрудники, у которых указан отдел.

## OUTER JOIN

**Возвращает все строки из одной или обеих таблиц, даже если нет совпадений.** Есть три типа:

### 1. LEFT OUTER JOIN (или просто LEFT JOIN)

**Возвращает все строки из левой таблицы и совпадающие из правой (NULL если нет совпадений).**

Пример:
```sql
SELECT employees.name, departments.name
FROM employees
LEFT JOIN departments ON employees.department_id = departments.id;
```
Результат: все сотрудники, даже те, у которых не указан отдел (в этом случае имя отдела будет NULL).

### 2. RIGHT OUTER JOIN (или просто RIGHT JOIN)

**Возвращает все строки из правой таблицы и совпадающие из левой (NULL если нет совпадений).**

Пример:
```sql
SELECT employees.name, departments.name
FROM employees
RIGHT JOIN departments ON employees.department_id = departments.id;
```
Результат: все отделы, даже те, в которых нет сотрудников (в этом случае имя сотрудника будет NULL).

### 3. FULL OUTER JOIN (или просто FULL JOIN)

**Возвращает все строки из обеих таблиц (NULL где нет совпадений).**

Пример:
```sql
SELECT employees.name, departments.name
FROM employees
FULL JOIN departments ON employees.department_id = departments.id;
```
Результат: все сотрудники и все отделы, с NULL где нет соответствий.

## Визуальное сравнение

Представим две таблицы:

**Таблица Employees:**
```
| id | name   | department_id |
|----|--------|---------------|
| 1  | Alice  | 101           |
| 2  | Bob    | 102           |
| 3  | Carol  | NULL          |
```

**Таблица Departments:**
```
| id | name     |
|----|----------|
| 101| Sales    |
| 102| Marketing|
| 103| HR       |
```

Результаты разных JOIN:

1. **INNER JOIN:**
   ```
   | Alice | Sales    |
   | Bob   | Marketing|
   ```

2. **LEFT JOIN:**
   ```
   | Alice | Sales    |
   | Bob   | Marketing|
   | Carol | NULL     |
   ```

3. **RIGHT JOIN:**
   ```
   | Alice | Sales    |
   | Bob   | Marketing|
   | NULL  | HR       |
   ```

4. **FULL JOIN:**
   ```
   | Alice | Sales    |
   | Bob   | Marketing|
   | Carol | NULL     |
   | NULL  | HR       |
   ```

## Когда что использовать?

- **INNER JOIN** - когда нужны только полные соответствия
- **LEFT JOIN** - когда нужно все из левой таблицы плюс совпадения справа
- **RIGHT JOIN** - когда нужно все из правой таблицы плюс совпадения слева
- **FULL JOIN** - когда нужно все данные из обеих таблиц
