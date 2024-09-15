create or replace view projects_category_transition
as
select
    sortby,
    split_part(data_category, '_', 2) as data_category,
    created_at,
    (created_at - INTERVAL '1 day')::date + INTERVAL '1 day' - EXTRACT(DOW FROM (created_at - INTERVAL '1 day')) * INTERVAL '1 day' AS weekly_monday,
    date_trunc('month', created_at)::date AS monthly_date,
    date_trunc('quarter', created_at)::date AS quarterly_date,
    date_trunc('year', created_at)::date AS yearly_date,
    avg(page) as page_average,
    avg(ranking) as ranking_average,
    avg(data_position) as position_average,
    count(*) as ranking_count,
    max(current_funding) as current_funding_max,
    avg(current_funding) as current_funding_average,
    max(current_supporters) as current_supporters_max,
    avg(current_supporters) as current_supporters_average,
    max(success_rate) as success_rate_max,
    avg(success_rate) as success_rate_average,
    max(remaining_days) as remaining_days_max,
    avg(remaining_days) as remaining_days_average
from projects as pj
group by sortby, data_category, created_at
order by sortby, data_category, created_at desc

