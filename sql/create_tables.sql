-- ============================================================
-- CREATE TABLES for Talent Match Intelligence Case Study
-- ============================================================

CREATE TABLE IF NOT EXISTS employee_analysis (
  employee_id VARCHAR PRIMARY KEY,
  fullname TEXT,
  gender TEXT,
  department TEXT,
  job_level TEXT,
  year INT,
  rating INT,
  avg_competency_score FLOAT,
  iq FLOAT,
  mbti TEXT,
  disc TEXT,
  first_word TEXT,
  second_word TEXT,
  years_experience FLOAT,
  leadership_score FLOAT,
  communication_score FLOAT,
  cognitive_score FLOAT
);

-- Optional index untuk mempercepat query
CREATE INDEX IF NOT EXISTS idx_rating ON employee_analysis(rating);
CREATE INDEX IF NOT EXISTS idx_department ON employee_analysis(department);
