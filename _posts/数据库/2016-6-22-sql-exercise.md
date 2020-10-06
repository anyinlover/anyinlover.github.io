---
layout: single
title: "sql练习"
subtitle: "取题sql-ex网站"
date: 2016-6-22
author: "Anyinlover"
category: 数据库
tags:
  - SQL
---

1. Find the model number, speed and hard drive capacity for all the PCs with prices below \$500.Result set: model, speed, hd.

   ```sql
   select model, speed, hd
   from PC
   where price<500;
   ```

2. List all printer makers. Result set: maker.

   ```sql
   select distinct maker
   from Product
   where type='Printer';
   ```

3. Find the model number, RAM and screen size of the laptops with prices over \$1000.

   ```sql
   select model, ram, screen
   from Laptop
   where price>1000;
   ```

4. Find all records from the Printer table containing data about color printers.

   ```sql
   select *
   from Printer
   where color='y';
   ```

5. Find the model number, speed and hard drive capacity of PCs cheaper than \$600 having a 12x or a 24x CD drive.

   ```sql
   select model, speed, hd
   from PC
   where price<600
   and cd in ('12x','24x');
   ```

6. For each maker producing laptops with a hard drive capacity of 10 Gb or higher, find the speed of such laptops. Result set: maker, speed.

   ```sql
   select distinct maker, speed
   from Product, Laptop
   where Product.model = Laptop.model
   and hd >= 10;
   ```

7. Find out the models and prices for all the products (of any type) produced by maker B.

   ```sql
   select Product.model, price
   from Product, PC
   where Product.model = PC.model
   and maker='B'
   union
   select Product.model, price
   from Product, Laptop
   where Product.model = Laptop.model
   and maker='B'
   union
   select Product.model, price
   from Product, Printer
   where Product.model = Printer.model
   and maker='B';
   ```

8. Find the makers producing PCs but not laptops.

   ```sql
   select distinct maker
   from Product
   where type='PC'
   and maker not in
   (select distinct maker
   from Product
   where type='Laptop')
   ```

9. Find the makers of PCs with a processor speed of 450 MHz or more. Result set: maker.

   ```sql
   select distinct maker
   from Product, PC
   where Product.model = PC.model
   and speed >= 450;
   ```

10. Find the printer models having the highest price. Result set: model, price.

    ```sql
    select model, price
    from Printer
    where price =
    (select max(price)
    from Printer);
    ```

11. Find out the average speed of PCs.

    ```sql
    select avg(speed)
    from PC;
    ```

12. Find out the average speed of the laptops priced over \$1000.

    ```sql
    select avg(speed)
    from Laptop
    where price > 1000;
    ```

13. Find out the average speed of the PCs produced by maker A.

    ```sql
    select avg(speed)
    from Product, PC
    where Product.model = PC.model
    and maker='A';
    ```

14. Get the makers who produce only one product type and more than one model. Output: maker, type.

    ```sql
    select distinct maker, type
    from Product
    where maker in
    (select maker
    from Product
    group by maker
    having count(distinct type)=1
    and count(distinct model)>1);
    ```

15. Get hard drive capacities that are identical for two or more PCs. Result set: hd.

    ```sql
    select hd
    from PC
    group by hd
    having count(*) > 1
    ```

16. Get pairs of PC models with identical speeds and the same RAM capacity. Each resulting pair should be displayed only once, i.e. (i, j) but not (j, i). Result set: model with the bigger number, model with the smaller number, speed, and RAM.

    ```sql
    select distinct p1.model, p2.model, p1.speed, p1.ram
    from PC p1, PC p2
    where p1.model > p2.model
    and p1.speed = p2.speed
    and p1.ram = p2.ram;
    ```

17. Get the laptop models that have a speed smaller than the speed of any PC. Result set: type, model, speed.

    ```sql
    select distinct type, Product.model, speed
    from Product, Laptop
    where Product.model = Laptop.model
    and speed <
    (select min(speed)
    from PC);
    ```

18. Find the makers of the cheapest color printers.Result set: maker, price.

    ```sql
    select distinct maker, price
    from Printer, Product
    where Product.model=Printer.model
    and color='y'
    and price=
    (select min(price)
    from Printer
    where color='y');
    ```

19. For each maker having models in the Laptop table, find out the average screen size of the laptops he produces. Result set: maker, average screen size.

    ```sql
    select maker, avg(screen)
    from Product, Laptop
    where Product.model = Laptop.model
    group by maker;
    ```

20. Find the makers producing at least three distinct models of PCs. Result set: maker, number of PC models.

    ```sql
    select maker, count(model)
    from Product
    where type='PC'
    group by maker
    having count(model) >= 3;
    ```

21. Find out the maximum PC price for each maker having models in the PC table. Result set: maker, maximum price.

    ```sql
    select maker, max(price)
    from Product, PC
    where Product.model = PC.model
    group by maker;
    ```

22. For each value of PC speed that exceeds 600 MHz, find out the average price of PCs with identical speeds. Result set: speed, average price.

    ```sql
    select speed, avg(price)
    from PC
    where speed > 600
    group by speed;
    ```

23. Get the makers producing both PCs having a speed of 750 MHz or higher and laptops with a speed of 750 MHz or higher. Result set: maker

    ```sql
    select distinct maker
    from Product, PC
    where Product.model = PC.model
    and speed >= 750
    and maker in
    (select distinct maker
    from Product, Laptop
    where Product.model = Laptop.model
    and speed >= 750)
    ```

24. List the models of any type having the highest price of all products present in the database.

    ```sql
    select distinct model
    from
    (select model, price
    from PC
    union all
    select model, price
    from Laptop
    union all
    select model, price
    from Printer) as union_model
    where price >= all
    (select price from (
    select price from PC
    union all
    select price from Laptop
    union all
    select price from Printer) as union_price);
    ```

25. Find the printer makers also producing PCs with the lowest RAM capacity and the highest processor speed of all PCs having the lowest RAM capacity. Result set: maker.

    ```sql
    select distinct maker
    from Product, PC
    where Product.model = PC.model
    and ram =
    (select min(ram)
    from PC)
    and speed =
    (select max(speed)
    from PC
    where ram =
    (select min(ram)
    from PC))
    and maker in
    (select maker
    from Product
    where type='Printer');
    ```

26. Find out the average price of PCs and laptops produced by maker A. Result set: one overall average price for all items.

    ```sql
    select avg(price)
    from
    (select price
    from Product, PC
    where Product.model = PC.model
    and maker = 'A'
    union all
    select price
    from Product, Laptop
    where Product.model = Laptop.model
    and maker='A'
    ) as union_price;
    ```

27. Find out the average hard disk drive capacity of PCs produced by makers who also manufacture printers.
    Result set: maker, average HDD capacity.

    ```sql
    select maker, avg(hd)
    from Product, PC
    where Product.model = PC.model
    and maker in
    (select distinct maker
    from Product
    where type='printer')
    group by maker;
    ```

28. Determine the average quantity of paint per square with an accuracy of two decimal places.

    ```sql
    select
    round(sum(ifnull(B_VOL,0))/count(distinct Q_ID),2)
    from utB right join utQ
    on utB.B_Q_ID = utQ.Q_ID;
    ```

29. Under the assumption that receipts of money (inc) and payouts (out) are registered not more than once a day for each collection point [i.e. the primary key consists of (point, date)], write a query displaying cash flow data (point, date, income, expense).
    Use Income_o and Outcome_o tables.

    ```sql
    select i.point, i.date, i.inc, o.out
    from Income_o i left join Outcome_o o
    on i.point = o.point
    and i.date = o.date
    union
    select o.point, o.date, i.inc, o.out
    from Outcome_o o left join Income_o i
    on o.point = i.point
    and o.date = i.date
    ```

30. Under the assumption that receipts of money (inc) and payouts (out) can be registered any number of times a day for each collection point [i.e. the code column is the primary key], display a table with one corresponding row for each operating date of each collection point.
    Result set: point, date, total payout per day (out), total money intake per day (inc).
    Missing values are considered to be NULL.

    ```sql
    select i.point, i.date, o.out, i.inc
    from
    (select point, date, sum(inc) as inc
    from Income
    group by point, date) i
    left join
    (select point, date, sum(out) as out
    from Outcome
    group by point, date) o
    on i.point = o.point
    and i.date = o.date
    union
    select o.point, o.date, o.out, i.inc
    from
    (select point, date, sum(out) as out
    from Outcome
    group by point, date) o
    left join
    (select point, date, sum(inc) as inc
    from Income
    group by point, date) i
    on o.point = i.point
    and o.date = i.date;
    ```

31. For ship classes with a gun caliber of 16 in. or more, display the class and the country.

    ```sql
    select class, country
    from Classes
    where bore >= 16;
    ```

32. One of the characteristics of a ship is one-half the cube of the calibre of its main guns (mw).
    Determine the average.

    ```sql
    select country, round(avg(power(bore,3)*0.5),2)
    from
    (select country, bore, name
    from Classes, Ships
    where Classes.class = Ships.class
    union
    select country, bore, ship
    from Classes, Outcomes
    where Classes.class = Outcomes.ship
    ) as new_ships
    group by country;
    ```

33. Get the ships sunk in the North Atlantic battle.
    Result set: ship.

    ```sql
    select ship
    from Outcomes
    where battle = 'North Atlantic'
    and result = 'sunk';
    ```

34. In accordance with the Washington Naval Treaty concluded in the beginning of 1922, it was prohibited to build battle ships with a displacement of more than 35 thousand tons.
    Get the ships violating this treaty (only consider ships for which the year of launch is known).
    List the names of the ships.

    ```sql
    select distinct name
    from Ships, Classes
    where Ships.class = Classes.class
    and type = 'bb'
    and displacement > 35000
    and launched >= 1922;
    ```

35. Find models in the Product table consisting either of digits only or Latin letters (A-Z, case insensitive) only.
    Result set: model, type.

    ```sql
    select model, type
    from Product
    where model regexp '^(([0-9]+)|([a-zA-Z]+))$';
    ```

36. List the names of lead ships in the database (including the Outcomes table).

    ```sql
    select name
    from Ships
    where name in
    (select class
    from Classes)
    union
    select ship
    from Outcomes
    where ship in
    (select class
    from Classes);
    ```

37. Find classes for which only one ship exists in the database (including the Outcomes table).

    ```sql
    select class
    from
    (select Classes.class, name
    from Classes, Ships
    where Classes.class = Ships.class
    union
    select class, ship as name
    from Classes, Outcomes
    where Classes.class = Outcomes.ship
    ) as full_ship
    group by class
    having count(*) = 1;
    ```

38. Find countries that ever had classes of both battleships (‘bb’) and cruisers (‘bc’).

    ```sql
    select distinct country
    from Classes
    where type='bb'
    and country in
    (select distinct country
    from Classes
    where type='bc');
    ```

39. Find the ships that "survived for future battles"; that is, after being damaged in a battle, they participated in another one, which occurred later.

    ```sql
    select distinct o2.ship from
    (select ship, battle, result, date
    from Outcomes, Battles
    where Outcomes.battle = Battles.name
    and result='damaged'
    ) as o1,
    (select ship, battle, result, date
    from Outcomes, Battles
    where Outcomes.battle = Battles.name
    ) as o2
    where o1.ship = o2.ship
    and o1.date < o2.date;
    ```

40. For the ships in the Ships table that have at least 10 guns, get the class, name, and country.

    ```sql
    select Ships.class, name, country
    from Ships left join Classes
    on Ships.class = Classes.class
    where numGuns >= 10;
    ```

41. For the PC in the PC table with the maximum code value, obtain all its characteristics (except for the code) and display them in two columns: name of the characteristic (title of the corresponding column in the PC table);its respective value.

    ```sql
    select 'model', model
    from PC
    where code=
    (select max(code)
    from PC)
    union
    select 'speed', speed
    from PC
    where code=
    (select max(code)
    from PC)
    union
    select 'ram', ram
    from PC
    where code=
    (select max(code)
    from PC)
    union
    select 'hd', hd
    from PC
    where code=
    (select max(code)
    from PC)
    union
    select 'cd', cd
    from PC
    where code=
    (select max(code)
    from PC)
    union
    select 'price', price
    from PC
    where code=
    (select max(code)
    from PC)
    ```

42. Find the names of ships sunk at battles, along with the names of the corresponding battles.

    ```sql
    select ship, battle
    from Outcomes
    where result='sunk';
    ```

43. Get the battles that occurred in years when no ships were launched into water.

    ```sql
    select name
    from Battles
    where year(date)
    not in
    (select launched
    from Ships
    where launched is not null);
    ```

44. Find all ship names beginning with the letter R.

    ```sql
    select name
    from Ships
    where name like 'R%'
    union
    select ship
    from Outcomes
    where ship like 'R%';
    ```

45. Find all ship names consisting of three or more words (e.g., King George V).
    Consider the words in ship names to be separated by single spaces, and the ship names to have no leading or trailing spaces.

    ```sql
    select name
    from Ships
    where name like '% % %'
    union
    select ship
    from Outcomes
    where ship like '% % %';
    ```

46. For each ship that participated in the Battle of Guadalcanal, get its name, displacement, and the number of guns.

    ```sql
    select distinct ship, displacement, numguns
    from Classes left join Ships
    on classes.class=ships.class
    right join Outcomes
    on Classes.class=ship
    or ships.name=ship
    where battle='Guadalcanal';
    ```

47. Number the rows of the Product table as follows: makers in descending order of number of models produced by them (for manufacturers producing an equal number of models, their names are sorted in ascending alphabetical order); model numbers in ascending order.
    Result set: row number as described above, manufacturer's name (maker), model.

    ```sql
    select count(*) num, t1.maker, t1.model
    from (
    select maker, model, c
    from Product
    join (
    select count(model) c, maker m
    from Product
    group by maker ) b1
    on maker = m) t1
    join
    (select maker, model, c
    from Product
    join (
    select count(model) c, maker m
    from Product
    group by maker ) b2
    on maker = m) t2
    on t2.c > t1.c
    or t2.c=t1.c and t2.maker<t1.maker
    or t2.c=t1.c and t2.maker=t1.maker and t2.model <= t1.model
    group by t1.maker, t1.model
    order by 1;
    ```

48. Find the ship classes having at least one ship sunk in battles.

    ```sql
    select distinct Classes.class
    from Classes, Ships, Outcomes
    where Classes.class = Ships.class
    and Ships.name = Outcomes.ship
    and Outcomes.result = 'sunk'
    union
    select distinct class
    from Classes, Outcomes
    where Classes.class = Outcomes.ship
    and Outcomes.result = 'sunk';
    ```

49. Find the names of the ships having a gun caliber of 16 inches (including ships in the Outcomes table).

    ```sql
    select name
    from Ships, Classes
    where Ships.class = Classes.class
    and bore = 16
    union
    select ship
    from Outcomes, Classes
    where Outcomes.ship = Classes.class
    and bore = 16;
    ```

50. Find the battles in which Kongo-class ships from the Ships table were engaged.

    ```sql
    select battle
    from Outcomes, Ships
    where Outcomes.ship = Ships.name
    and Ships.class = 'Kongo';
    ```
