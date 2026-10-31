-- Q1: WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS spend_over_500
FROM orders WHERE status='completed' AND order_amount > 500
GROUP BY customer_id HAVING SUM(order_amount) > 8000;

-- Q2: revenue by city and product category
SELECT c.city, p.category, ROUND(SUM(o.order_amount),2) AS revenue
FROM orders o JOIN customers c USING(customer_id) JOIN products p USING(product_id)
WHERE o.status='completed'
GROUP BY c.city,p.category ORDER BY c.city,p.category;

-- Q3: products above average catalogue price
SELECT product_id, product_name, price_inr FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products);

-- Q4: customers without completed orders
SELECT c.customer_id FROM customers c LEFT JOIN orders o
ON c.customer_id=o.customer_id AND o.status='completed'
WHERE o.order_id IS NULL;

-- Q5: CTE for above-average customer spend
WITH customer_spend AS (
 SELECT customer_id, SUM(order_amount) AS total_spend FROM orders
 WHERE status='completed' GROUP BY customer_id
), average_spend AS (SELECT AVG(total_spend) AS avg_spend FROM customer_spend)
SELECT customer_id,total_spend FROM customer_spend,average_spend
WHERE total_spend > avg_spend;

-- Q6: ranking window functions within city
WITH spend AS (
 SELECT c.customer_id,c.city,SUM(o.order_amount) AS total_spend
 FROM customers c JOIN orders o USING(customer_id)
 WHERE o.status='completed' GROUP BY c.customer_id,c.city
)
SELECT *, RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total_spend DESC) AS row_num
FROM spend;

-- Q7: daily running total
WITH daily AS (
 SELECT substr(order_ts,1,10) AS day,SUM(order_amount) AS revenue
 FROM orders WHERE status='completed' GROUP BY substr(order_ts,1,10)
)
SELECT day,revenue,SUM(revenue) OVER(ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue
FROM daily ORDER BY day;
