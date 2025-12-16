/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package exempluinterfataprolog;

import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author Irina
 */
public class ExempluInterfataProlog {
    static final int PORT=5003;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here

        ConexiuneProlog cxp;

        try {
            final Fereastra fereastra=new Fereastra("Exemplu Interfata Prolog");

            cxp=new ConexiuneProlog(PORT,fereastra);
            fereastra.setConexiune(cxp);
            fereastra.setVisible(true);
            fereastra.addWindowListener(new WindowAdapter() {
                public void windowClosing(WindowEvent e) {
                    try {
                        fereastra.conexiune.opresteProlog();
                        fereastra.conexiune.expeditor.gata=true;
                    } catch (InterruptedException ex) {
                        Logger.getLogger(ExempluInterfataProlog.class.getName()).log(Level.SEVERE, null, ex);
                    }
                }
            });
        } catch (IOException ex) {
            Logger.getLogger(ExempluInterfataProlog.class.getName()).log(Level.SEVERE, null, ex);
            System.out.println("Eroare clasa initiala");
        } catch (InterruptedException ex) {
            Logger.getLogger(ExempluInterfataProlog.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
}
