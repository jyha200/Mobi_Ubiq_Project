package org.tensorflow.lite.examples.detection;

import android.app.AlertDialog;
import android.content.Intent;
import android.os.Bundle;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.view.View;
import android.widget.ArrayAdapter;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.ListView;

public class MenuActivity extends AppCompatActivity {
  private Button returnButton;

  @Override
  protected void onCreate(@Nullable Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    setContentView(R.layout.tfe_od_activity_menu);
    String[] LIST_MENU = getStrings();

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
        builder.setMessage(strText + " message");
        builder.setNeutralButton("Confirm", null);
        builder.create().show();
      }
    });
  }

  String[] getStrings() {
    String[] ret = {"LIST1", "LIST2", "LIST3"} ;
    return ret;
  }
}