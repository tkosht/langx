create or replace view project_details_latest
AS
select * from project_details pd
where (pd.project_id, pd.created_at) in
	(
		select project_id, max(created_at)
		from project_details
		group by project_id
	)
order by project_id
