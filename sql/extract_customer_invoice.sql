
select 
c.customer_no, 
c.customer_name, 
inv.invoice_date::timestamp::date, 
--inv.invoice_due_date::timestamp::date, 
status.status,
1 as constant
from customers c
inner join accounts a on c.id = a.customer_id
left join invoices inv on inv.account_id = a.id
left join invoice_status status on inv.invoice_status_id = status.id