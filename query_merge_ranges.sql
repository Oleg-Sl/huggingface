WITH grouped_sessions AS (
  SELECT 
    user_id,
    start_date,
    end_date,
    MAX(end_date) OVER (
      PARTITION BY user_id 
      ORDER BY start_date 
      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS max_end_date_so_far
  FROM sessions
),
merged_groups AS (
  SELECT 
    user_id,
    start_date,
    end_date,
    CASE 
      WHEN start_date <= LAG(max_end_date_so_far) OVER (PARTITION BY user_id ORDER BY start_date) 
      THEN 0 
      ELSE 1 
    END AS is_new_group
  FROM grouped_sessions
),
group_ids AS (
  SELECT 
    user_id,
    start_date,
    end_date,
    SUM(is_new_group) OVER (PARTITION BY user_id ORDER BY start_date) AS group_id
  FROM merged_groups
)
SELECT 
  user_id,
  MIN(start_date) AS start_date,
  MAX(end_date) AS end_date
FROM group_ids
GROUP BY user_id, group_id
ORDER BY user_id, start_date;
