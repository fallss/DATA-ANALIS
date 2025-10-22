-- ============================================================
-- Success Pattern Analysis
-- ============================================================

WITH success_factors AS (
  SELECT 
    rating,
    AVG(iq) AS avg_iq,
    AVG(avg_competency_score) AS avg_comp,
    AVG(leadership_score) AS avg_leadership,
    AVG(communication_score) AS avg_communication,
    COUNT(*) AS total
  FROM employee_analysis
  GROUP BY rating
)
SELECT
  rating,
  ROUND(avg_iq, 2) AS iq,
  ROUND(avg_comp, 2) AS competency,
  ROUND(avg_leadership, 2) AS leadership,
  ROUND(avg_communication, 2) AS communication,
  total
FROM success_factors
ORDER BY rating DESC;
