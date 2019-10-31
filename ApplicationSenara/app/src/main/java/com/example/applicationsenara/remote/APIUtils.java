package com.example.applicationsenara.remote;

public class APIUtils {

    private APIUtils(){
    }

    public static final String API_URL = "http://10.0.2.2:8000/";  // Put the IP address of the backend server here, currently set to a localserver

    public static FileService getFileService() {
        return RetrofitClient.getClient(API_URL).create(FileService.class);
    }
}
