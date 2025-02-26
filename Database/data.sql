INSERT INTO "vehicle" ("id", "plate", "model", "color", "brand", "year", "owner", "created_at", "updated_at", "deleted_at") VALUES ('1', 'AVI2214', 'Mustang', 'Red', 'Ford', 1964, 'John Doe', '2021-01-01', '2021-01-01', NULL);

INSERT INTO "vehicle" ("id", "plate", "model", "color", "brand", "year", "owner", "created_at", "updated_at", "deleted_at") 
VALUES ('2', 'AQY6388', 'Gol', 'Black', 'Volkswagen', 2000, 'Jane Doe', '2021-02-03', '2021-02-03', NULL);

INSERT INTO "vehicle" ("id", "plate", "model", "color", "brand", "year", "owner", "created_at", "updated_at", "deleted_at") VALUES ('3', 'ABC1D25', 'Corola', 'White', 'Toyota', 2005, 'John Doe', '2021-03-01', '2021-03-01', '2023-10-05');

INSERT INTO "vehicle" ("id", "plate", "model", "color", "brand", "year", "owner", "created_at", "updated_at", "deleted_at") VALUES ('4', 'ABC1D26', 'Civic', 'Blue', 'Honda', 2008, 'Jane Doe', '2021-04-01', '2021-04-03', NULL);

INSERT INTO "vehicle_log" ("id", "vehicle_id", "severity", "type", "description", "created_at", "updated_at", "deleted_at") VALUES ('1', '1', 'DANGER', 'STOLEN', 'desc', '2021-05-03', '2022-07-03', NULL);

INSERT INTO "vehicle_log" ("id", "vehicle_id", "severity", "type", "description", "created_at", "updated_at", "deleted_at") VALUES ('2', '2', 'DANGER', 'FUGITIVE_OWNER', 'desc', '2022-04-01', '2022-04-03', NULL);

INSERT INTO "vehicle_log" ("id", "vehicle_id", "severity", "type", "description", "created_at", "updated_at", "deleted_at") VALUES ('3', '2', 'DANGER', 'OTHER', 'desc', '2021-04-01', '2021-04-03', '2022-04-03');