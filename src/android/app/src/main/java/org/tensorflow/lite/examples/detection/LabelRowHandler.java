package com.example.helloworld;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.text.TextUtils;
import android.util.Log;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class LabelRowHandler extends SQLiteOpenHelper{

    private HashMap<Integer, String> labelIndex = new HashMap<Integer, String>(){
        {
            put(2, "machine");
            put(3, "handwash");
            put(4, "nowater");
            put(5, "bleach_O");
            put(6, "bleach_X");
            put(7, "dryer_O");
            put(8, "dryer_X");
            put(9, "wring_O");
            put(10, "wring_X");
            put(11, "sun");
            put(12, "shade");
            put(13, "iron_O");
            put(14, "iron_X");
            put(15, "dryclean_O");
            put(16, "dryclean_X");
        }
    };

    private static final String DB_NAME = "laundryDB";
    private static final int DB_VERSION = 1;
    private static final String TABLE_NAME = "careLabel";
    private static final String ID_COL = "id";
    private static final String CLOTH_COL = "userClothType";
    private static final String DESC_COL1 = "machine";
    private static final String DESC_COL2 = "handwash";
    private static final String DESC_COL3 = "nowater";
    private static final String DESC_COL4 = "bleach_O";
    private static final String DESC_COL5 = "bleach_X";
    private static final String DESC_COL6 = "dryer_O";
    private static final String DESC_COL7 = "dryer_X";
    private static final String DESC_COL8 = "wring_O";
    private static final String DESC_COL9 = "wring_X";
    private static final String DESC_COL10 = "sun";
    private static final String DESC_COL11 = "shade";
    private static final String DESC_COL12 = "iron_O";
    private static final String DESC_COL13 = "iron_X";
    private static final String DESC_COL14 = "dryclean_O";
    private static final String DESC_COL15 = "dryclean_X";


    public LabelRowHandler(Context context) {
        super(context, DB_NAME, null, DB_VERSION);
    }

    @Override
    public void onCreate(SQLiteDatabase db) {

        String query = "CREATE TABLE IF NOT EXISTS " + TABLE_NAME + " ("
                + ID_COL + " INTEGER PRIMARY KEY AUTOINCREMENT, "
                + CLOTH_COL + " INT,"
                + DESC_COL1 + " INT,"
                + DESC_COL2 + " INT,"
                + DESC_COL3 + " INT,"
                + DESC_COL4 + " INT,"
                + DESC_COL5 + " INT,"
                + DESC_COL6 + " INT,"
                + DESC_COL7 + " INT,"
                + DESC_COL8 + " INT,"
                + DESC_COL9 + " INT,"
                + DESC_COL10 + " INT,"
                + DESC_COL11 + " INT,"
                + DESC_COL12 + " INT,"
                + DESC_COL13 + " INT,"
                + DESC_COL14 + " INT,"
                + DESC_COL15 + " INT);";

        db.execSQL(query);

    }

    public void addNewUserDataLabelRow(String userCloth, String[] labelArray) {

        SQLiteDatabase db = this.getWritableDatabase();

        ContentValues values = new ContentValues();

        String deleteQuery = "DELETE FROM " + TABLE_NAME + " WHERE " + CLOTH_COL + "= '" + userCloth + "';";
        db.execSQL(deleteQuery); // testing purpose

        List<Integer> inputLabelList = new ArrayList<>();

        values.put(CLOTH_COL, "cl");

        for(int i = 0; i < labelArray.length; i++) {
            values.put(labelArray[i], 31);
        }

        db.insert(TABLE_NAME, null, values);

        db.close();
    }

    public List<String> getUserDataLabelRow(String userCloth) {

        List<String> labelList = new ArrayList<>();
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("SELECT * FROM " + TABLE_NAME + " WHERE " + CLOTH_COL + "= '" + userCloth + "';", null);
        if (cursor != null) {
            cursor.moveToFirst();
        }

        int columnExists;

        for(int i = 0; i < 15; i++) {
            columnExists = cursor.getInt(i+2);
            if (columnExists > 0) {
                labelList.add(labelIndex.get(i+2));
            }
        }

        return labelList;

    }

    public List<String> getClothes() {

        List<String> clothesList = new ArrayList<>();
        SQLiteDatabase db = this.getReadableDatabase();
        Cursor cursor = db.rawQuery("SELECT "+ CLOTH_COL + " FROM " + TABLE_NAME + "';", null);
        if (cursor != null) {
            cursor.moveToFirst();
        }

        for (cursor.moveToFirst(); !cursor.isAfterLast(); cursor.moveToNext()) {
            clothesList.add(cursor.getString(1));
            // do what you need with the cursor here
        }

        return clothesList;

    }

    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        // this method is called to check if the table exists already.
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_NAME);
        onCreate(db);
    }

}

// private Button button1;
// private Button button2;
// private DBHandler dbHandler;

// button1 = findViewById(R.id.helloworld_button);
// button2 = findViewById(R.id.button2);

// dbHandler = new DBHandler(MainActivity.this);

// button1.setOnClickListener(new View.OnClickListener() {
//     @Override
//     public void onClick(View v) {

//         // below line is to get data from all edit text fields.
//         String clothType = "Water_Laundry";
//         String clothLabel = "Wash it using washing machine";

//         // validating if the text fields are empty or not.
//         if (clothType.isEmpty() && clothLabel.isEmpty()) {
//             Toast.makeText(MainActivity.this, "Please enter all the data..", Toast.LENGTH_SHORT).show();
//             return;
//         }

//         String[] arrTemp = {"machine", "handwash"};
//         dbHandler.addNewUserDataLabelRow("cl", arrTemp);


//         // after adding the data we are displaying a toast message.
//         Toast.makeText(MainActivity.this, "new row has been added.", Toast.LENGTH_SHORT).show();
//     }
// });

// button2.setOnClickListener(new View.OnClickListener() {
//     @Override
//     public void onClick(View v) {
//         List temp;
//         temp = dbHandler.getUserDataLabelRow("cl");

//         String result = TextUtils.join(",", temp);


//         Toast.makeText(MainActivity.this, result, Toast.LENGTH_SHORT).show();
//     }
// });