DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

CREATE TYPE SEVERITY AS ENUM ('CRITICAL', 'DANGER', 'WARNING', 'INFO');

CREATE TYPE LOG_TYPE AS ENUM ('STOLEN', 'FUGITIVE_OWNER', 'OTHER');

CREATE TABLE "vehicle"(
    "id" SERIAL NOT NULL,
    "plate" VARCHAR(255) NOT NULL,
    "model" VARCHAR(255) NOT NULL,
    "color" VARCHAR(255) NOT NULL,
    "brand" VARCHAR(255) NOT NULL,
    "year" BIGINT NOT NULL,
    "owner" VARCHAR(255) NOT NULL,
    "created_at" DATE NOT NULL,
    "updated_at" DATE NOT NULL,
    "deleted_at" DATE NULL
);
ALTER TABLE
    "vehicle" ADD PRIMARY KEY("id");
ALTER TABLE
    "vehicle" ADD CONSTRAINT "vehicle_plate_unique" UNIQUE("plate");
CREATE TABLE "vehicle_log"(
    "id" SERIAL NOT NULL,
    "vehicle_id" BIGINT NOT NULL,
    "severity" SEVERITY NOT NULL,
    "type" LOG_TYPE NOT NULL,
    "description" VARCHAR(255) NOT NULL,
    "created_at" DATE NOT NULL,
    "updated_at" DATE NOT NULL,
    "deleted_at" DATE NULL
);
ALTER TABLE
    "vehicle_log" ADD PRIMARY KEY("id");
ALTER TABLE
    "vehicle_log" ADD CONSTRAINT "vehicle_log_vehicle_id_foreign" FOREIGN KEY("vehicle_id") REFERENCES "vehicle"("id");