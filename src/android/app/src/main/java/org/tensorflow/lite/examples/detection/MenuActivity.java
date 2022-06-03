package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.content.DialogInterface;
import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.text.TextUtils;
import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.logging.Logger;

//import org.tensorflow.lite.examples.detection.env.Logger;

public class MenuActivity extends AppCompatActivity {
  private Button returnButton;

//  private static final Logger LOGGER = new Logger();

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_od_activity_menu);

    //db create and insert testing data

    String[] arrTemp = {"machine", "dryer_O"};
    String[] arrTemp2 = {"nowater", "dryer_O", "wring_X"};
    String[] arrTemp3 = {"handwash", "shade"};

    CameraActivity.dbHandler.addNewUserDataLabelRow("pink T-shirt", arrTemp);
    CameraActivity.dbHandler.addNewUserDataLabelRow("black jean", arrTemp2);
    CameraActivity.dbHandler.addNewUserDataLabelRow("baseball jumper", arrTemp3);

    String[] LIST_MENU = getClothIdxs();

    ArrayAdapter adapter = new ArrayAdapter(this, android.R.layout.simple_list_item_1, LIST_MENU);

    ListView listview = (ListView) findViewById(R.id.cloth_list);
    listview.setAdapter(adapter);
    listview.setOnItemClickListener(new AdapterView.OnItemClickListener() {
      @Override
      public void onItemClick(AdapterView parent, View v, int position, long id) {
        // get TextView's Text.
        String strText = (String) parent.getItemAtPosition(position);
        AlertDialog.Builder builder = new AlertDialog.Builder(MenuActivity.this);
        builder.setTitle(strText);
        String[] labels = getCareLabels(strText);
        List<String> msg_list = new ArrayList<>();
        for (int i = 0 ; i < labels.length ; i++) {
          String guide = PictureActivity.reference_guide.get(labels[i]);
          msg_list.add(guide);
        }

        CharSequence[] items = msg_list.toArray(new String[msg_list.size()]);
        builder.setItems(items, new DialogInterface.OnClickListener() {
          @Override
          public void onClick(DialogInterface dialogInterface, int i) {
            String text = items[i].toString();
            Toast.makeText(getApplicationContext(), text, Toast.LENGTH_SHORT).show();
          }
        });
        builder.setNeutralButton("Confirm", null);
        builder.create().show();
      }
    });
  }

  String[] getClothIdxs() {
    List clothesList;
    clothesList = CameraActivity.dbHandler.getUserDataClothes();
    return (String[]) clothesList.toArray(new String[clothesList.size()]);

//    String[] ret = {"pink T-shirt", "black jean", "baseball jumper"};
//    return ret;
  }

  String[] getCareLabels(String cloth_idx) {
    List labelList;
    labelList = CameraActivity.dbHandler.getUserDataLabelRow(cloth_idx);
    return (String[]) labelList.toArray(new String[labelList.size()]);

//    String[] ret = {"machine", "bleach_O", "dryer_X", "wring_O","sun","iron_X"} ;
//    return ret;
  }
}