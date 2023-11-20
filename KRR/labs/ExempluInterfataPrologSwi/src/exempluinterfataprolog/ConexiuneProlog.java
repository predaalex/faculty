/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package exempluinterfataprolog;

import java.io.IOException;
import java.io.InputStream;
import java.io.PipedOutputStream;
import java.io.PrintStream;
import java.net.ServerSocket;
import java.net.Socket;

/**
 *
 * @author Irina
 */
public class ConexiuneProlog {
    final String caleExecutabilSicstus="C:\\Program Files\\swipl\\bin\\swipl-win.exe";
    
    final String nume_fisier="exemplu_prolog.pl";
    
    final String scop="inceput.";    
   

    Process procesSicstus;
    ExpeditorMesaje expeditor;
    CititorMesaje cititor;
    Fereastra fereastra;
    int port;
    
    
    public Fereastra getFereastra(){
        return fereastra;
    }
    
    public ConexiuneProlog(int _port, Fereastra _fereastra) throws IOException, InterruptedException{
        InputStream processIs, processStreamErr;
        port=_port;
        fereastra=_fereastra;
     
        //acces la mediul curent de rulare
        ServerSocket servs=new ServerSocket(port);
        
        cititor=new CititorMesaje(this,servs);
        cititor.start();
        expeditor=new ExpeditorMesaje(cititor);
        expeditor.start();
        
        
        Runtime rtime= Runtime.getRuntime();
        
        String comanda=caleExecutabilSicstus+" -g "+scop+" "+nume_fisier+" -- "+port;
        
        procesSicstus=rtime.exec(comanda);
        
        //InputStream-ul din care citim ce scrie procesul
        processIs=procesSicstus.getInputStream();
        //stream-ul de eroare
        processStreamErr=procesSicstus.getErrorStream();     

    }
    
    void opresteProlog() throws InterruptedException{
        PipedOutputStream pos= this.expeditor.getPipedOutputStream();
        PrintStream ps=new PrintStream(pos);
        ps.println("exit.");
        ps.flush();
    }
}



