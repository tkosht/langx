from fastapi import FastAPI, Query, Body, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
import pandas
from io import BytesIO
from app.controller.apicontroller import ApiController


app = FastAPI()
app.mount("/front", StaticFiles(directory="frontend/dist"), name="front")
app.mount(
    "/static", StaticFiles(directory="frontend/dist/static"), name="static"
)  # must be '/static'


@app.get("/hello")
async def hello():
    print({"Hello": "World!!!"})
    return {"Hello": "World!!!"}


@app.post("/post")
async def echo(type: str, name: str = Query(None), body: dict = Body(None)):
    print({"type": type, "name": name, "body": body})
    return {"type": type, "name": name, "body": body}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    code = 0
    statuses = ["success", "error"]
    detail = f"to upload the file '{file.filename}'"
    columns = []
    data = []
    try:
        print("uploaded:", file.filename)
        contents = await file.read()
        # print("contents:", contents.decode('utf-8')[:100])
        df = pandas.read_csv(
            BytesIO(contents), encoding="utf-8", parse_dates=True, index_col=0
        )
        print("df:", df.columns)
        apc = ApiController()
        predicted_df = apc.predict_trained(df)
        columns = list(predicted_df.columns)
        data = predicted_df.values.tolist()

    except Exception as e:
        print("Exception:", e)
        code = 1
        detail = str(e)
    finally:
        print("[webapi] done.")
        return {
            "code": code,
            "status": statuses[code],
            "detail": detail,
            "columns": columns,
            "data": data,
        }


@app.get("/")
async def redirect_index_html():
    return RedirectResponse("front/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
