package com.example.applicationsenara.remote;
import com.example.applicationsenara.model.FileInfo;
import com.example.applicationsenara.model.FinalResults;
import com.example.applicationsenara.model.ImageToProcess;
import com.example.applicationsenara.model.NearbyWatermeters;

import okhttp3.MultipartBody;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.Multipart;
import retrofit2.Call;
import retrofit2.http.POST;
import retrofit2.http.PUT;
import retrofit2.http.Part;
import retrofit2.http.Path;
import retrofit2.http.Query;

public interface FileService {

    // HTTP calls made to the server for the handling of thed data

    @Multipart
    @POST("image/") // Uploads the image
    Call<FileInfo> upload(@Part MultipartBody.Part file);

    @GET("processing/")
    Call<ImageToProcess> process( // Starts the image processing
            @Query("image") String imageLocation,
            @Query("lat") double latitude,
            @Query("long") double longitude);

    @GET("processing/near/") // Gets the nearest 5 watermeters
    Call<NearbyWatermeters> get_nearby_watermeters(
            @Query("lat") double latitude,
            @Query("long") double longitude);

    @POST("/processing/update/{image}") // Updates the results after they are reviewed by the user
    Call<FinalResults> updateValues(@Path("image") String imageName, @Body FinalResults finalResults);
}
