create or replace view projects_view
AS
select
    pj.*,
    replace(data_category, 'category_', '') as category,
    case
        when pj.current_supporters = 0 then null
        else pj.current_funding / pj.current_supporters
    end as funding_by_supporter,
    (created_at - INTERVAL '1 day')::date + INTERVAL '1 day' - EXTRACT(DOW FROM (created_at - INTERVAL '1 day')) * INTERVAL '1 day' AS weekly_monday,
    date_trunc('month', created_at)::date AS monthly_date,
    date_trunc('quarter', created_at)::date AS quarterly_date,
    date_trunc('year', created_at)::date AS yearly_date
from projects as pj

