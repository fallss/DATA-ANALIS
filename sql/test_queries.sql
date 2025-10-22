-- Hitung total data dan employee unik
SELECT COUNT(*) AS total_rows, COUNT(DISTINCT employee_id) AS unique_employees
FROM employee_analysis;

-- Lihat distribusi rating
SELECT rating, COUNT(*) AS jumlah
FROM employee_analysis
GROUP BY rating
ORDER BY rating DESC;

-- Rata-rata IQ dan kompetensi per rating
SELECT 
  rating, 
  ROUND(AVG(iq), 2) AS avg_iq,
  ROUND(AVG(avg_competency_score), 2) AS avg_competency
FROM employee_analysis
GROUP BY rating
ORDER BY rating DESC;

-- Cek data tahun terbaru
SELECT DISTINCT year FROM employee_analysis ORDER BY year DESC;
