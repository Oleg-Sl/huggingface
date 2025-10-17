WITH numbered_actions AS (
  SELECT 
    *,
    LAG(action_number) OVER (PARTITION BY user_id ORDER BY action_date) AS prev_action_number
  FROM user_actions
),
session_starts AS (
  SELECT 
    *,
    CASE 
      -- Первое действие пользователя
      WHEN prev_action_number IS NULL THEN 1
      -- Сброс нумерации (текущий номер меньше или равен предыдущему)
      -- WHEN action_number <= prev_action_number
      WHEN action_number <= prev_action_number OR EXTRACT(EPOCH FROM (action_date - prev_action_date)) >= 1800 THEN 1
      THEN 1
      ELSE 0v
    END AS is_session_start
  FROM numbered_actions
)
SELECT 
  *,
  SUM(is_session_start) OVER (PARTITION BY user_id ORDER BY action_date) AS session_id
FROM session_starts
ORDER BY user_id, action_date;







WITH numbered_actions AS (
  SELECT 
    *,
    LAG(action_number) OVER (PARTITION BY user_id ORDER BY action_date) AS prev_action_number
  FROM user_actions
),
session_starts AS (
  SELECT 
    *,
    CASE 
      -- Первое действие пользователя
      WHEN prev_action_number IS NULL THEN 1
      -- Перепад номера на 2 или более единиц
      WHEN action_number - prev_action_number >= 2 THEN 1
      -- Или если номер уменьшился (сброс нумерации)
      WHEN action_number <= prev_action_number THEN 1
      -- WHEN action_number - prev_action_number >= 2 OR EXTRACT(EPOCH FROM (action_date - prev_action_date)) >= 1800 THEN 1
      ELSE 0
    END AS is_session_start
  FROM numbered_actions
)
SELECT 
  *,
  SUM(is_session_start) OVER (PARTITION BY user_id ORDER BY action_date) AS session_id
FROM session_starts
ORDER BY user_id, action_date;
