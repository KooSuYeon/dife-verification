from locust import HttpUser, task, between
import os

class WebsiteUser(HttpUser):
    wait_time = between(0.5, 2)  # 요청 간 대기 시간

    @task
    def check_similarity(self):
        with open("dataset/full_verification.jpeg", "rb") as f:
            self.client.post(
                "/check_similarity/",
                files={"file": ("dataset/full_verification.jpeg", f, "image/jpeg")}
            )
